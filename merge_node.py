"""Merge de deux modeles MiniMax H3, entrees INT8 acceptees.

Architecture : tout se passe au niveau des TENSEURS, avant instanciation du
modele (meme principe que ModelMerge Krea 2 + Quant).

- Un poids INT8 natif ComfyUI (int8_tensorwise, avec ou sans ConvRot) est
  dequantifie par comfy_kitchen vers son vrai poids : la rotation ConvRot est
  defaite, donc deux modeles quantifies differemment se fusionnent quand meme
  correctement.
- La fusion se fait en float32 (torch.lerp), puis le poids est requantifie
  dans le format du modele de base (unet_1) : meme ConvRot, meme taille de
  groupe, meme type d'echelle.
- Un ratio de 1.0 garde le tenseur de unet_1 tel quel, un ratio de 0.0 recopie
  celui de unet_2 tel quel (avec son echelle) : aucun arrondi supplementaire.
  Un echange de blocs facon Hybrid Loader reste donc exact.

- LoRA optionnels (3 max) : appliques sur le poids fusionne en float, juste
  avant la requantification. Le MODEL et le fichier sauvegarde les incluent.

unet_2 est lu tenseur par tenseur : un seul modele complet en RAM. Les fichiers
sont lus sans mmap et la sauvegarde ecrit tenseur par tenseur, pour que la RAM
ne contienne jamais deux copies du modele.
Autonome : ne depend que de ComfyUI, comfy_kitchen et safetensors.
"""

import gc
import json
import os
import struct
import time
import zlib

import torch

import comfy.lora
import comfy.memory_management
import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

try:
    from comfy.lora_convert import convert_lora as _convert_lora
except Exception:
    _convert_lora = None

INT8_FORMAT = "int8_tensorwise"
LOG = "[H3Merge]"

# Cles hors blocs DiT traitees comme entree/sortie du modele.
H3_IO_PREFIXES = (
    "video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.",
    "adaln_t_table", "rope.",
)
# Chemin texte : projection des embeddings Qwen3-VL et token refiner.
H3_TEXT_PREFIXES = ("condition_proj.", "token_refiner.")
# Base adaLN stockee a part par certains checkpoints "pruned" (x = mean + c @ basis).
# Le modele ComfyUI ne la lit pas ; elle decrit le repere de la table adaLN.
H3_BASIS_EXTRAS = ("adaln_basis", "adaln_mean")
# Ecart relatif au-dela duquel deux adaln_t_table ne sont pas le meme repere.
ADALN_TABLE_TOLERANCE = 1e-3
# Matrices des blocs DiT et du token refiner quantifiables depuis bf16/fp32.
# L'adaLN, les projections d'entree/sortie, condition_proj et les normes restent
# en pleine precision.
H3_QUANT_LAYERS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
INT8_CONVROT_GROUPSIZE = 256
# Sortie "int8 HQ" : comme les checkpoints H3 "HQ" / "skip edges", le chemin du
# texte et les blocs du bord restent en bf16.
HQ_FIRST_BLOCKS = 2
HQ_LAST_BLOCKS = 3
# Sorties W6A8 / W4A8 : poids 6 ou 4 bits par groupes de 16, echelles FP8,
# ConvRot 256, calcul INT8 (meme layout ComfyUI, la largeur se deduit du poids).
W6A8_FORMAT = "w6a8_int8"
W4A8_FORMAT = "asym_w4a8_int8"
W6A8_GROUP_SIZE = 16
# Formats "bas bits" : exigent in_features multiple de 256, sinon la couche reste en INT8.
LOW_FORMATS = {"w6a8": W6A8_FORMAT, "w4a8": W4A8_FORMAT}
LOW_NAMES = {"w6a8": "W6A8", "w4a8": "W4A8"}
# Cles annexes d'un poids quantifie, a retirer quand on change de format.
QUANT_COMPANIONS = ("weight_scale", "input_scale", "weight_s_rel", "weight_s_channel",
                    "weight_codebook", "weight_correction")
LOW_CONVROT_GROUPSIZE = 256


# ----------------------------------------------------------------------
# Logique pure (sans torch) : nommage, groupes, ratios
# ----------------------------------------------------------------------

def _short(key, prefix):
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def _block_index(short):
    """Index du bloc DiT principal (blocks.N.*), None ailleurs.

    token_refiner.blocks.N.* n'est PAS un bloc DiT : il appartient au chemin texte.
    """
    if not short.startswith("blocks."):
        return None
    try:
        return int(short.split(".", 2)[1])
    except (IndexError, ValueError):
        return None


def _ratio_for(short, cfg):
    """Part de unet_1 pour ce tenseur. cfg : dict des ratios et bornes.

    Une valeur < 0 herite du ratio global. Le reglage adaLN, s'il est actif,
    prime sur le groupe de blocs pour blocks.N.adaln_proj.* dans sa plage.
    """
    def resolve(value):
        return cfg["global"] if value < 0.0 else value

    idx = _block_index(short)
    if idx is not None:
        rest = short.split(".", 2)[2] if short.count(".") >= 2 else ""
        if (cfg["adaln"] >= 0.0 and cfg["adaln_start"] <= idx <= cfg["adaln_end"]
                and rest.startswith("adaln_proj.")):
            return cfg["adaln"]
        if idx <= cfg["early_end"]:
            return resolve(cfg["early"])
        if idx <= cfg["mid_end"]:
            return resolve(cfg["mid"])
        return resolve(cfg["late"])
    if short.startswith(H3_TEXT_PREFIXES):
        return resolve(cfg["text"])
    if short.startswith(H3_IO_PREFIXES):
        return resolve(cfg["in_out"])
    return cfg["global"]


def _is_adaln_family(short):
    """Tenseurs exprimes dans le repere de la table adaLN (forme courbe)."""
    if short == "adaln_t_table" or short in H3_BASIS_EXTRAS:
        return True
    if short.startswith("final_layer.adaln_proj."):
        return True
    if _block_index(short) is None or short.count(".") < 2:
        return False
    return short.split(".", 2)[2].startswith("adaln_proj.")


def _h3_quantizable(short):
    """Poids d'une matrice qkv / out_proj / fc1 / fc2 (blocs DiT ou token refiner)."""
    if not short.endswith(".weight"):
        return False
    parts = short[:-len(".weight")].split(".")
    if len(parts) == 4 and parts[0] == "blocks" and parts[1].isdigit():
        tail = parts[2:]
    elif len(parts) == 5 and parts[:2] == ["token_refiner", "blocks"] and parts[2].isdigit():
        tail = parts[3:]
    else:
        return False
    return ".".join(tail) in H3_QUANT_LAYERS


def _h3_hq_keep(short, n_blocks):
    """Matrice gardee en bf16 par la sortie "int8 HQ" : condition_proj, le token
    refiner et les blocs 0-1 et les 3 derniers (47-49 sur 50)."""
    if short == "condition_proj.weight":
        return True
    if not _h3_quantizable(short):
        return False
    if short.startswith("token_refiner."):
        return True
    i = int(short.split(".")[1])
    return i < HQ_FIRST_BLOCKS or i >= n_blocks - HQ_LAST_BLOCKS


def _int8_params(conf):
    """(convrot, convrot_groupsize) d'une config comfy_quant int8_tensorwise."""
    params = conf.get("params", {})
    if not isinstance(params, dict):
        params = {}
    convrot = bool(conf.get("convrot", params.get("convrot", False)))
    groupsize = int(conf.get("convrot_groupsize", params.get("convrot_groupsize", 256)))
    return convrot, groupsize


def _companion_keys(base):
    """Cles portees par un poids quantifie, en plus de <base>.weight."""
    return (f"{base}.weight_scale", f"{base}.comfy_quant", f"{base}.input_scale")


# ----------------------------------------------------------------------
# (De)quantification via comfy_kitchen
# ----------------------------------------------------------------------

def _conf_from_tensor(tensor):
    return json.loads(tensor.cpu().numpy().tobytes())


def _conf_to_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _int8_layout():
    from comfy.quant_ops import TensorWiseINT8Layout, get_layout_class
    if get_layout_class("TensorWiseINT8Layout") is None or not hasattr(TensorWiseINT8Layout, "quantize"):
        raise RuntimeError("INT8 indisponible : comfy_kitchen absent ou trop ancien. "
                           "Mettez ComfyUI et ses dependances a jour.")
    return TensorWiseINT8Layout


def _dequantize(qdata, scale, conf, device):
    """Vrai poids float32 (rotation ConvRot defaite) d'un poids int8_tensorwise."""
    layout = _int8_layout()
    convrot, groupsize = _int8_params(conf)
    q = qdata.to(device=device)
    params = layout.Params(
        scale=scale.to(device=device), orig_dtype=torch.float32, orig_shape=tuple(q.shape),
        convrot=convrot, convrot_groupsize=groupsize,
    )
    return layout.dequantize(q, params).to(dtype=torch.float32)


def _quantize_like(weight, ref_conf, ref_scale, device, stochastic_rounding=0):
    """Requantifie dans le format du poids de reference (unet_1)."""
    layout = _int8_layout()
    convrot, groupsize = _int8_params(ref_conf)
    per_channel = convrot or ref_scale.numel() > 1
    qdata, params = layout.quantize(
        weight.to(device=device, dtype=torch.float32), is_weight=True,
        per_channel=per_channel, convrot=convrot, convrot_groupsize=groupsize,
        stochastic_rounding=stochastic_rounding,
    )
    scale = params.scale
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(float(scale))
    # Garde la convention d'ecriture de unet_1 (dtype et forme de l'echelle).
    scale = scale.to(dtype=ref_scale.dtype)
    if scale.numel() == ref_scale.numel():
        scale = scale.reshape(ref_scale.shape)
    return qdata.to(dtype=torch.int8).cpu(), scale.cpu()


def _quantize_int8(weight, device):
    """Quantifie un poids float en INT8 natif ComfyUI (ConvRot si in_features
    est multiple de 256, sinon par canal) -> (qdata, scale, config)."""
    layout = _int8_layout()
    convrot = int(weight.shape[1]) % INT8_CONVROT_GROUPSIZE == 0
    qdata, params = layout.quantize(
        weight.to(device=device, dtype=torch.float32), is_weight=True, per_channel=True,
        convrot=convrot, convrot_groupsize=INT8_CONVROT_GROUPSIZE, stochastic_rounding=0,
    )
    conf = {"format": INT8_FORMAT}
    if convrot:
        conf.update(convrot=True, convrot_groupsize=INT8_CONVROT_GROUPSIZE)
    return qdata.to(dtype=torch.int8).cpu(), params.scale.cpu(), conf


def _low_available(out_format):
    fmt = LOW_FORMATS[out_format]
    try:
        from comfy.quant_ops import QUANT_ALGOS, get_layout_class
        if fmt not in QUANT_ALGOS:
            return False, f"format {fmt} absent de QUANT_ALGOS (ComfyUI trop ancien ?)"
        name = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
        cls = get_layout_class(name)
        if cls is None or not hasattr(cls, "quantize"):
            return False, f"layout {name} indisponible (comfy_kitchen absent ou trop ancien)"
        return True, None
    except Exception as e:
        return False, str(e)


def _low_fits(shape):
    """W6A8 et W4A8 exigent une matrice 2D dont in_features est multiple de 256."""
    return len(shape) == 2 and int(shape[1]) % LOW_CONVROT_GROUPSIZE == 0


def _quantize_grouped(weight, device, bits):
    """Poids groupes de ComfyUI : 6 bits uniformes (w6a8_int8) ou 4 bits avec table
    Lloyd-Max (asym_w4a8_int8) -> (qdata, {cle annexe: tenseur}, config comfy_quant)."""
    fmt = W6A8_FORMAT if bits == 6 else W4A8_FORMAT
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class
    layout = get_layout_class(QUANT_ALGOS[fmt]["comfy_tensor_layout"])
    qdata, params = layout.quantize(
        weight.to(device=device, dtype=torch.float32), group_size=W6A8_GROUP_SIZE,
        convrot_groupsize=LOW_CONVROT_GROUPSIZE, symmetric=True,
        scale_dtype=torch.float8_e4m3fn, codebook=(bits == 4), stochastic_rounding=0, bits=bits,
    )
    if getattr(params, "correction", None) is not None:
        raise RuntimeError(f"{fmt} : correction asymetrique inattendue (non chargeable par ComfyUI).")
    extras = {"weight_s_rel": params.scale.cpu(), "weight_s_channel": params.s_channel.cpu()}
    if getattr(params, "codebook", None) is not None:
        extras["weight_codebook"] = params.codebook.cpu()
    conf = {"format": fmt, "group_size": W6A8_GROUP_SIZE,
            "convrot_groupsize": LOW_CONVROT_GROUPSIZE}
    return qdata.cpu(), extras, conf


def _quantize_w6a8(weight, device):
    return _quantize_grouped(weight, device, 6)


def _quantize_w4a8(weight, device):
    return _quantize_grouped(weight, device, 4)


def _quantize_low(weight, device, out_format):
    """W6A8 ou W4A8 -> (qdata, {cle annexe: tenseur}, config comfy_quant)."""
    if out_format == "w6a8":
        return _quantize_w6a8(weight, device)
    return _quantize_w4a8(weight, device)


def _set_quantized(sd, key, qdata, extras, conf):
    """Ecrit un poids quantifie et ses cles annexes, en retirant celles de l'ancien format."""
    full_base = key[:-len(".weight")]
    for name in QUANT_COMPANIONS:
        sd.pop(f"{full_base}.{name}", None)
    sd[key] = qdata
    for name, tensor in extras.items():
        sd[f"{full_base}.{name}"] = tensor
    sd[f"{full_base}.comfy_quant"] = _conf_to_tensor(conf)


# ----------------------------------------------------------------------
# Lecture / ecriture .safetensors sans mmap ni copie complete
# ----------------------------------------------------------------------

_ST_ITEMSIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2,
                "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
                "F8_E4M3": 1, "F8_E5M2": 1}
_ST_TORCH = {"F64": "float64", "I64": "int64", "F32": "float32", "I32": "int32",
             "F16": "float16", "BF16": "bfloat16", "I16": "int16", "I8": "int8",
             "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn",
             "F8_E5M2": "float8_e5m2"}
_IO_CHUNK = 64 * 1024 * 1024


def _st_dtype(name):
    dtype = getattr(torch, _ST_TORCH.get(name, ""), None)
    if dtype is None:
        raise ValueError(f"type de tenseur non gere : {name}")
    return dtype


def _st_name(tensor):
    for name, attr in _ST_TORCH.items():
        if getattr(torch, attr, None) == tensor.dtype:
            return name
    raise ValueError(f"type de tenseur non gere : {tensor.dtype}")


def _tensor_bytes(tensor):
    """Octets d'un tenseur CPU, sans copie (vue uint8)."""
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous().reshape(-1)
    return memoryview(t.view(torch.uint8).numpy())


def _tensor_from_buffer(buf, name, shape):
    if len(buf) == 0:
        return torch.empty(shape, dtype=_st_dtype(name))
    return torch.frombuffer(buf, dtype=_st_dtype(name)).reshape(shape)


class _SliceInfo:
    def __init__(self, info):
        self.info = info

    def get_shape(self):
        return list(self.info["shape"])

    def get_dtype(self):
        return self.info["dtype"]


class _SafetensorsFile:
    """.safetensors lu par fichier classique (sans mmap) : chaque tenseur est
    copie en RAM a la demande et rien ne reste mappe dans le processus."""

    def __init__(self, path):
        self.fh = open(path, "rb")
        size = struct.unpack("<Q", self.fh.read(8))[0]
        header = json.loads(self.fh.read(size))
        self._meta = header.pop("__metadata__", None) or {}
        self.header = header
        self.base = 8 + size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fh.close()

    def keys(self):
        return list(self.header)

    def metadata(self):
        return dict(self._meta)

    def get_slice(self, key):
        return _SliceInfo(self.header[key])

    def get_tensor(self, key):
        info = self.header[key]
        start, end = info["data_offsets"]
        buf = bytearray(end - start)
        view, done = memoryview(buf), 0
        self.fh.seek(self.base + start)
        while done < len(buf):
            got = self.fh.readinto(view[done:done + _IO_CHUNK])
            if not got:
                raise IOError(f"fichier tronque ({key})")
            done += got
        return _tensor_from_buffer(buf, info["dtype"], info["shape"])


def _write_safetensors(sd, path, metadata=None):
    """Ecrit un .safetensors tenseur par tenseur, sans copie du state_dict.

    (safetensors.save_file peut, selon sa version, recopier tout le modele en
    RAM avant d'ecrire.) Tenseurs ranges par taille d'element decroissante
    pour garder les donnees alignees, comme le fait safetensors.
    """
    entries = []
    for key, tensor in sd.items():
        name = _st_name(tensor)
        entries.append((-_ST_ITEMSIZE[name], key, name, tensor))
    entries.sort(key=lambda e: (e[0], e[1]))
    header, offset = {}, 0
    for _, key, name, tensor in entries:
        nbytes = _ST_ITEMSIZE[name]
        for d in tensor.shape:
            nbytes *= int(d)
        header[key] = {"dtype": name, "shape": [int(d) for d in tensor.shape],
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for _, key, name, tensor in entries:
            data = _tensor_bytes(tensor)
            if len(data) != header[key]["data_offsets"][1] - header[key]["data_offsets"][0]:
                raise RuntimeError(f"taille inattendue pour {key}")
            for i in range(0, len(data), _IO_CHUNK):
                fh.write(data[i:i + _IO_CHUNK])
            del data
    os.replace(tmp, path)


def _free_unused_cache():
    """Libere les resultats en cache que ce run n'utilise pas (le MODEL du merge
    precedent et ses copies Load LoRA / ModelSampling...).

    ComfyUI fait ce nettoyage entre deux nodes ; le merge etant souvent le
    premier node relance, il n'aurait lieu qu'apres lui, avec l'ancien modele
    encore en RAM pendant tout le merge. Meme appel que ComfyUI avant d'allouer
    de la memoire. Retourne les octets liberes (estimation de ComfyUI).
    """
    try:
        target = int(comfy.model_management.total_ram * 1024 * 1024)
        freed = comfy.memory_management.extra_ram_release(target) or 0
    except Exception:
        freed = 0
    gc.collect()
    try:
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    return freed


def _ram():
    try:
        import psutil
        proc = psutil.Process().memory_info().rss / 1024 ** 3
        return f"RAM processus {proc:.1f} Go, systeme {psutil.virtual_memory().percent:.0f} %"
    except Exception:
        return "RAM ?"


# ----------------------------------------------------------------------
# Lecture paresseuse de unet_2
# ----------------------------------------------------------------------

class _LazySource:
    """unet_2 lu tenseur par tenseur, avec sa config de quantification."""

    def __init__(self, handle):
        self.f = handle
        self.keys = list(handle.keys())
        self.key_set = set(self.keys)
        meta = handle.metadata() or {}
        self.layers = {}
        if "_quantization_metadata" in meta:
            try:
                self.layers = json.loads(meta["_quantization_metadata"]).get("layers", {}) or {}
            except (ValueError, TypeError, AttributeError):
                self.layers = {}
        self.prefix = comfy.model_detection.unet_prefix_from_state_dict(dict.fromkeys(self.keys))
        self.by_short = {_short(k, self.prefix): k for k in self.keys}
        self._conf_cache = {}

    def has(self, short):
        return short in self.by_short

    def shape(self, short):
        return tuple(self.f.get_slice(self.by_short[short]).get_shape())

    def tensor(self, short):
        return self.f.get_tensor(self.by_short[short])

    def conf(self, base_short):
        """Config comfy_quant du poids <base_short>.weight, ou None (non quantifie)."""
        if base_short in self._conf_cache:
            return self._conf_cache[base_short]
        conf = None
        key = f"{base_short}.comfy_quant"
        if key in self.by_short:
            conf = _conf_from_tensor(self.tensor(key))
        else:
            full_base = self.by_short.get(f"{base_short}.weight", "")[:-len(".weight")]
            for candidate in (full_base, base_short):
                if candidate in self.layers:
                    conf = self.layers[candidate]
                    break
        self._conf_cache[base_short] = conf
        return conf


class _SelfSource:
    """unet_2 = None : unet_1 sert de source, avec tous les ratios a 1.0. Rien
    n'est melange ; seuls les LoRA et le format de sortie s'appliquent."""

    def __init__(self, sd, prefix):
        self.sd = sd
        self.keys = list(sd)
        self.by_short = {_short(k, prefix): k for k in self.keys}

    def has(self, short):
        return short in self.by_short

    def shape(self, short):
        return tuple(self.sd[self.by_short[short]].shape)

    def tensor(self, short):
        return self.sd[self.by_short[short]]

    def conf(self, base_short):
        key = f"{base_short}.comfy_quant"
        return _conf_from_tensor(self.tensor(key)) if key in self.by_short else None


# ----------------------------------------------------------------------
# LoRA : appliques sur les poids fusionnes en float, avant requantification
# ----------------------------------------------------------------------

_LORA_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight", ".lora.down.weight",
                       ".lora_linear_layer.down.weight")


def _lora_key_map(sd, prefix, metadata):
    """Noms de modules LoRA -> cle du state_dict (conventions de comfy.lora pour H3).

    Meme table que model_lora_keys_unet + branche MiniMaxH3, construite sans
    instancier le modele. Les tetes PDD empilees recoivent un offset (tete 0).
    """
    key_map = {}
    for k in sd:
        short = _short(k, prefix)
        if short.endswith(".weight"):
            base = short[:-len(".weight")]
            for alias in (base, f"diffusion_model.{base}", f"model.diffusion_model.{base}",
                          f"transformer.{base}", f"lora_unet_{base.replace('.', '_')}"):
                key_map[alias] = k
        elif not short.endswith((".comfy_quant", ".weight_scale", ".input_scale")):
            key_map[short] = k
            key_map[f"diffusion_model.{short}"] = k
    pdd_steps = 1
    try:
        pdd_steps = int(json.loads((metadata or {}).get("config", "{}"))
                        .get("transformer", {}).get("pdd_steps", 1))
    except Exception:
        pdd_steps = 1
    if pdd_steps > 1:
        for name in ("final_layer.video_out.weight", "final_layer.audio_out.weight"):
            k = prefix + name
            if k in sd:
                head = (k, (0, 0, sd[k].shape[0] // pdd_steps))
                key_map.update({alias: head for alias, v in key_map.items() if v == k})
    return key_map


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _patch_mismatch(patch, shape):
    """Raison du rejet si le patch ne colle pas a la forme du tenseur, sinon None.

    comfy.lora ignore ces patches avec un simple message d'erreur : on les
    ecarte avant la passe pour pouvoir le dire clairement.
    """
    v, offset = patch[1], patch[3]
    shape = tuple(int(d) for d in shape)
    if offset is not None:
        shape = shape[:offset[0]] + (int(offset[2]),) + shape[offset[0] + 1:]
    weights = getattr(v, "weights", None)
    if getattr(v, "name", None) == "lora" and weights is not None \
            and weights[3] is None and weights[5] is None:
        up, down = weights[0], weights[1]
        rows, inner = int(up.shape[0]), _numel(down.shape[1:])
        if _numel(up.shape[1:]) != int(down.shape[0]) or rows * inner != _numel(shape):
            return f"LoRA {rows}x{inner} vs {shape}"
    elif isinstance(v, tuple) and len(v) == 2 and v[0] == "diff":
        diff = v[1][0]
        pad = len(v[1]) > 1 and isinstance(v[1][1], dict) and v[1][1].get("pad_weight")
        if not pad and tuple(int(d) for d in diff.shape) != shape:
            return f"diff {tuple(diff.shape)} vs {shape}"
    return None


def _load_loras(sd, prefix, metadata, specs):
    """Charge les LoRA -> ({cle courte: [patches]}, infos). N'applique rien."""
    active = [(n, float(s)) for n, s in specs if n and n != "None" and s != 0]
    if not active:
        return {}, []
    key_map = _lora_key_map(sd, prefix, metadata)
    targets, infos = {}, []
    for name, strength in active:
        path = folder_paths.get_full_path("loras", name)
        if path is None:
            raise ValueError(f"LoRA introuvable : {name}")
        try:
            data = comfy.utils.load_torch_file(path, safe_load=True)
            if _convert_lora is not None:
                try:
                    data = _convert_lora(data)
                except Exception:
                    pass
            modules = sum(1 for k in data if k.endswith(_LORA_DOWN_SUFFIXES))
            patch_dict = comfy.lora.load_lora(data, key_map)
        except Exception as e:
            raise ValueError(f"LoRA {name} non charge : {e}") from e
        applied, rejected = 0, []
        for k, v in patch_dict.items():
            key, offset, function = (k, None, None) if not isinstance(k, tuple) else \
                (k[0], k[1] if len(k) > 1 else None, k[2] if len(k) > 2 else None)
            if key not in sd:
                rejected.append(f"{_short(key, prefix)} (absent du modele)")
                continue
            patch = (strength, v, 1.0, offset, function)
            reason = _patch_mismatch(patch, sd[key].shape)
            if reason is not None:
                rejected.append(f"{_short(key, prefix)} ({reason})")
                continue
            targets.setdefault(_short(key, prefix), []).append(patch)
            applied += 1
        del data
        msg = f"{LOG} LoRA {name} (force {strength:g}) : {applied} tenseurs cibles"
        if modules > applied + len(rejected):
            msg += (f", {modules - applied - len(rejected)} module(s) sans equivalent dans ce "
                    "modele (voir 'lora key not loaded')")
        if rejected:
            msg += f", {len(rejected)} ignore(s) : " + ", ".join(rejected[:4])
            if len(rejected) > 4:
                msg += ", ..."
        print(msg + ".", flush=True)
        if applied == 0:
            raise ValueError(f"LoRA {name} : aucun patch applicable a ce modele "
                             "(LoRA pour une autre architecture ?). Rien n'a ete modifie.")
        infos.append({"name": name, "strength": strength, "tensors": applied,
                      "rejected": len(rejected)})
    return targets, infos


def _apply_lora(weight, patches, key):
    """Patches LoRA sur un poids float32 deja sur le device de calcul."""
    return comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=torch.float32)


def _lora_seed(key):
    """Graine fixe par tenseur pour l'arrondi stochastique (resultat reproductible)."""
    return (zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF) or 1


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class ModelMergeMiniMaxH3Int8:
    CATEGORY = "advanced/model_merging/model_specific"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    DESCRIPTION = ("Merge deux modeles MiniMax H3, INT8 ConvRot ou bf16 en entree. "
                   "Les poids INT8 sont dequantifies, fusionnes puis requantifies au "
                   "format de unet_1. Ratio 1.0 / 0.0 = copie exacte du tenseur. "
                   "LoRA optionnels appliques au modele fusionne (et sauvegardes avec).")

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        strength = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                              "tooltip": "Force du LoRA, applique au modele fusionne "
                                         "(avant requantification INT8)."})
        ratio = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Part de unet_1. 1.0 = 100% unet_1, "
                                      "0.0 = 100% unet_2, 0.5 = moyenne."})
        override = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                              "tooltip": "-1 = herite du ratio global. "
                                         "Sinon part de unet_1 pour ce groupe."})
        return {
            "required": {
                "unet_1": (unets, {"tooltip": "Modele de base : son format de quantification "
                                              "est conserve."}),
                "unet_2": (["None"] + unets, {
                    "tooltip": "None = pas de merge : unet_1 seul, avec ses LoRA et le format de "
                               "sortie choisi (les ratios sont ignores)."}),
                "ratio_global": ratio,
                "ratio_in_out": override,
                "ratio_text": override,
                "ratio_blocks_early": override,
                "ratio_blocks_mid": override,
                "ratio_blocks_late": override,
                "early_end": ("INT", {"default": 16, "min": 0, "max": 255,
                    "tooltip": "Dernier bloc DiT du groupe early (inclus). 16 = blocs 0-16."}),
                "mid_end": ("INT", {"default": 33, "min": 0, "max": 255,
                    "tooltip": "Dernier bloc du groupe mid (inclus). 33 = blocs 17-33, "
                               "le reste va dans late."}),
                "ratio_adaln": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "-1 = desactive. Sinon part de unet_1 pour blocks.N.adaln_proj "
                               "dans la plage adaln_start..adaln_end (prime sur les groupes). "
                               "Ex. hybride FL2VA + Ref2VA : ratio_global 1, ratio_adaln 0, "
                               "plage 30-49."}),
                "adaln_start": ("INT", {"default": 30, "min": 0, "max": 255,
                    "tooltip": "Premier bloc (inclus) concerne par ratio_adaln."}),
                "adaln_end": ("INT", {"default": 49, "min": 0, "max": 255,
                    "tooltip": "Dernier bloc (inclus) concerne par ratio_adaln."}),
                "save": (["disable", "enabled"], {"default": "disable"}),
                "filename_prefix": ("STRING", {"default": "merged/H3_merge"}),
            },
            "optional": {
                "lora_1": (loras,), "strength_1": strength,
                "lora_2": (loras,), "strength_2": strength,
                "lora_3": (loras,), "strength_3": strength,
                "output_format": (["bf16", "int8", "int8 HQ", "w6a8", "w6a8 HQ", "w4a8", "w4a8 HQ"],
                                  {"default": "int8",
                    "tooltip": "Format du modele fusionne, quel que soit le format des entrees. "
                               "int8 : INT8 ConvRot. w6a8 : poids 6 bits ConvRot (groupes de 16), "
                               "plus petit que int8. w4a8 : poids 4 bits (table Lloyd-Max, "
                               "groupes de 16), calcul INT8. bf16 : "
                               "rien de quantifie. Les variantes HQ gardent en bf16 le chemin du "
                               "texte (condition_proj, token refiner) et les blocs 0-1 et 47-49, "
                               "comme les versions HQ / skip edges. Seules les matrices qkv / "
                               "out_proj / fc1 / fc2 sont quantifiees depuis bf16 ; l'adaLN, les "
                               "entrees/sorties et les normes gardent leur format de unet_1."}),
            },
        }

    # ------------------------------------------------------------------
    # Verifications
    # ------------------------------------------------------------------

    @staticmethod
    def _quant_bases(shorts_with_conf):
        return {s[:-len(".comfy_quant")] for s in shorts_with_conf if s.endswith(".comfy_quant")}

    def _check_h3(self, shorts, label):
        if "video_patch_proj.weight" not in shorts or "condition_proj.weight" not in shorts:
            raise ValueError(f"{label} n'est pas un modele MiniMax H3 (video_patch_proj / "
                             "condition_proj introuvables).")

    # ------------------------------------------------------------------
    # Passe unique
    # ------------------------------------------------------------------

    def _merge(self, sd, prefix, src, cfg, device, lora_targets=None, out_format="int8"):
        t0 = time.time()
        a_short = {_short(k, prefix): k for k in sd.keys()}
        self._check_h3(a_short, "unet_1")
        self._check_h3(src.by_short, "unet_2")
        if ("adaln_t_table" in a_short) != src.has("adaln_t_table"):
            raise ValueError("unet_1 et unet_2 n'ont pas la meme forme adaLN (table de courbes "
                             "baked vs adaLN dense). Ces deux modeles ne se fusionnent pas "
                             "tenseur a tenseur : convertissez-les d'abord au meme format.")

        # Forme courbe : chaque poids adaLN est exprime dans le repere de la table
        # de son checkpoint. On ne les melange que si les deux tables sont le
        # meme repere ; sinon tout l'adaLN vient en bloc d'un seul modele.
        adaln_lock, table_diff = None, None
        if "adaln_t_table" in a_short:
            ta = sd[a_short["adaln_t_table"]].to(dtype=torch.float32)
            tb = src.tensor("adaln_t_table").to(dtype=torch.float32)
            if tuple(ta.shape) != tuple(tb.shape):
                table_diff = float("inf")
            else:
                table_diff = float((ta - tb).norm()) / max(float(ta.norm()), 1e-12)
            if table_diff > ADALN_TABLE_TOLERANCE:
                adaln_lock = 0.0 if _ratio_for("adaln_t_table", cfg) <= 0.0 else 1.0
                print(f"{LOG} adaln_t_table differentes (ecart {table_diff:.3g}) : reperes adaLN "
                      f"incompatibles, tout l'adaLN vient de {'unet_2' if adaln_lock == 0.0 else 'unet_1'} "
                      "(ratio_in_out = 0 pour prendre celui de unet_2). ratio_adaln est ignore.",
                      flush=True)
            else:
                print(f"{LOG} adaln_t_table identiques (ecart {table_diff:.3g}) : adaLN fusionnable.",
                      flush=True)

        a_quant = self._quant_bases(a_short)
        companions = {c for base in a_quant for c in _companion_keys(base)}

        # 1. Plan + verifications AVANT toute modification.
        lora_targets = lora_targets or {}
        # "xxx HQ" = format xxx, avec le texte et les blocs du bord gardes en bf16.
        out_hq = out_format.endswith(" HQ")
        fmt = out_format[:-len(" HQ")] if out_hq else out_format
        out_low = fmt in LOW_FORMATS
        out_bf16 = fmt == "bf16"
        quantize_float = fmt in ("int8", "w6a8", "w4a8")
        n_blocks = 1 + max((_block_index(x) for x in a_short if _block_index(x) is not None),
                           default=-1)

        def to_bf16_for(short, a_is_q):
            """Couche INT8 de unet_1 a sortir en bf16."""
            return a_is_q and (out_bf16 or (out_hq and _h3_hq_keep(short, n_blocks)))
        n_low_skip = 0
        plan, problems, extras_kept = [], [], []
        for short, key in a_short.items():
            if short in companions:
                continue
            r = _ratio_for(short, cfg)
            if adaln_lock is not None and _is_adaln_family(short):
                r = adaln_lock
            lora = lora_targets.get(short)
            need_b = r < 1.0
            base = short[:-len(".weight")] if short.endswith(".weight") else None
            a_is_q = base is not None and base in a_quant
            # Sortie W6A8 / W4A8 : toute couche INT8 de unet_1 est reconvertie, meme
            # gardee (sauf si sa forme ne s'y prete pas : elle reste en INT8).
            # Sortie bf16 (ou couche HQ) : la couche INT8 de unet_1 est remise en bf16.
            to_bf16 = to_bf16_for(short, a_is_q)
            to_low = out_low and a_is_q and not to_bf16 and _low_fits(tuple(sd[key].shape))
            if out_low and a_is_q and not to_bf16 and not to_low:
                n_low_skip += 1
            # Matrice bf16/fp32 de unet_1 a quantifier (sortie int8, w6a8 ou w4a8).
            float_q = (quantize_float and not a_is_q and _h3_quantizable(short)
                       and len(sd[key].shape) == 2
                       and not (out_hq and _h3_hq_keep(short, n_blocks)))
            if not need_b and not lora and not to_low and not to_bf16 and not float_q:
                continue
            if a_is_q:
                conf = _conf_from_tensor(sd[a_short[f"{base}.comfy_quant"]])
                if conf.get("format") != INT8_FORMAT:
                    problems.append(f"{short}: format unet_1 non supporte ({conf.get('format')})")
                    continue
                if f"{base}.weight_scale" not in a_short:
                    problems.append(f"{short}: echelle INT8 absente de unet_1")
                    continue
            elif sd[key].dtype in (torch.int8, torch.uint8) and short.endswith(".weight"):
                problems.append(f"{short}: poids entier sans config comfy_quant dans unet_1 "
                                "(format INT8 non natif : convertissez-le au format ComfyUI)")
                continue
            b_conf = None
            if need_b and not src.has(short):
                if "." not in short:
                    # Tenseur annexe de premier niveau (ex. adaln_basis) : garde.
                    extras_kept.append(short)
                    if not lora:
                        continue
                    need_b = False
                else:
                    problems.append(f"{short}: absent de unet_2")
                    continue
            if need_b:
                a_shape = tuple(sd[key].shape)
                b_shape = src.shape(short)
                if a_shape != b_shape:
                    problems.append(f"{short}: forme {a_shape} vs {b_shape}")
                    continue
                b_conf = src.conf(base) if base is not None else None
                if b_conf is not None and b_conf.get("format") != INT8_FORMAT:
                    problems.append(f"{short}: format unet_2 non supporte ({b_conf.get('format')})")
                    continue
                if (b_conf is None and short.endswith(".weight")
                        and src.f.get_slice(src.by_short[short]).get_dtype() in ("I8", "U8")):
                    problems.append(f"{short}: poids entier sans config comfy_quant dans unet_2 "
                                    "(format INT8 non natif : convertissez-le au format ComfyUI)")
                    continue
                if b_conf is not None and not src.has(f"{base}.weight_scale"):
                    problems.append(f"{short}: echelle INT8 absente de unet_2")
                    continue
            plan.append((short, key, base, r if need_b else 1.0, a_is_q, b_conf, need_b, lora,
                         float_q))

        # Un poids present seulement dans unet_2 = architecture differente
        # (ex. QKV separes) : il serait ignore sans bruit, on refuse.
        for short in src.by_short:
            if short.endswith(".weight") and short not in a_short:
                problems.append(f"{short}: present seulement dans unet_2 (architecture differente)")

        if problems:
            shown = "\n  ".join(problems[:8])
            more = f"\n  ... et {len(problems) - 8} autre(s)" if len(problems) > 8 else ""
            raise ValueError(f"Modeles incompatibles ({len(problems)} tenseur(s)) :\n  {shown}{more}\n"
                             "Rien n'a ete modifie ni sauvegarde.")

        # Repere adaLN pris de unet_2 : la base annexe de unet_1 ne le decrit plus.
        dropped = []
        if adaln_lock == 0.0:
            for name in H3_BASIS_EXTRAS:
                if name in a_short and not src.has(name):
                    del sd[a_short.pop(name)]
                    dropped.append(name)
            extras_kept = [x for x in extras_kept if x not in dropped]
            plan = [p for p in plan if p[0] not in dropped]

        only_a = sum(1 for s in a_short if s not in companions and not src.has(s))
        only_b = sum(1 for s in src.by_short if s not in a_short
                     and not s.endswith((".comfy_quant", ".weight_scale", ".input_scale")))

        # 2. Application : merge -> LoRA -> requantification, un tenseur a la fois.
        n_copy_b = n_requant = n_float = n_lora = n_low = n_newq = n_bf16 = 0
        try:
            pbar = comfy.utils.ProgressBar(len(plan))
        except Exception:
            pbar = None

        def b_float(short, base, b_conf):
            if b_conf is not None:
                return _dequantize(src.tensor(short), src.tensor(f"{base}.weight_scale"), b_conf, device)
            # copy=True : le LoRA s'ajoute en place, jamais dans le tenseur source.
            return src.tensor(short).to(device=device, dtype=torch.float32, copy=True)

        for short, key, base, r, a_is_q, b_conf, need_b, lora, float_q in plan:
            if pbar is not None:
                pbar.update(1)

            if a_is_q:
                scale_key = a_short[f"{base}.weight_scale"]
                conf_key = a_short[f"{base}.comfy_quant"]
                a_conf = _conf_from_tensor(sd[conf_key])
                a_scale = sd[scale_key]

                to_bf16 = to_bf16_for(short, True)
                if to_bf16 or (out_low and _low_fits(tuple(sd[key].shape))):
                    if not need_b:
                        w = _dequantize(sd[key], a_scale, a_conf, device)
                    elif r <= 0.0:
                        w = b_float(short, base, b_conf)
                    else:
                        w = torch.lerp(_dequantize(sd[key], a_scale, a_conf, device),
                                       b_float(short, base, b_conf), 1.0 - r)
                    if lora:
                        w = _apply_lora(w, lora, key)
                        n_lora += 1
                    input_scale = a_short.get(f"{base}.input_scale")
                    if input_scale is not None:
                        sd.pop(input_scale, None)
                    if to_bf16:
                        sd[key] = w.to(dtype=torch.bfloat16).cpu()
                        sd.pop(scale_key, None)
                        sd.pop(conf_key, None)
                        n_bf16 += 1
                    else:
                        _set_quantized(sd, key, *_quantize_low(w, device, fmt))
                        n_low += 1
                    del w
                    continue

                # Merge INT8 : exactement le chemin de la 1.0.1.
                conf = a_conf
                if not need_b:
                    qdata, scale = sd[key], a_scale
                elif r <= 0.0 and b_conf is not None:
                    # Copie exacte de unet_2 : octets, echelle et config d'origine.
                    qdata, scale, conf = src.tensor(short), src.tensor(f"{base}.weight_scale"), b_conf
                    sd[conf_key] = _conf_to_tensor(b_conf)
                    if not lora:
                        n_copy_b += 1
                else:
                    w = b_float(short, base, b_conf) if r <= 0.0 else torch.lerp(
                        _dequantize(sd[key], a_scale, a_conf, device),
                        b_float(short, base, b_conf), 1.0 - r)
                    qdata, scale = _quantize_like(w, a_conf, a_scale, device)
                    del w
                    n_requant += 1

                if lora:
                    # LoRA integre comme ComfyUI le fait : sur le poids INT8 deja
                    # fusionne, puis requantification a arrondi stochastique. Un
                    # arrondi au plus proche deforme le LoRA (il est amplifie quand
                    # le merge tombe a mi-chemin de la grille INT8, vers 0.5).
                    w = _apply_lora(_dequantize(qdata, scale, conf, device), lora, key)
                    qdata, scale = _quantize_like(w, conf, scale, device,
                                                  stochastic_rounding=_lora_seed(key))
                    del w
                    n_lora += 1
                sd[key] = qdata
                sd[scale_key] = scale
            else:
                orig = sd[key]
                if need_b:
                    w = b_float(short, base, b_conf) if r <= 0.0 else torch.lerp(
                        orig.to(device=device, dtype=torch.float32), b_float(short, base, b_conf), 1.0 - r)
                    merged_t = w.to(dtype=orig.dtype).cpu()
                    del w
                    n_float += 1
                else:
                    merged_t = orig
                if lora:
                    # copy=True : le LoRA s'ajoute en place, jamais dans le tenseur source.
                    w = _apply_lora(merged_t.to(device=device, dtype=torch.float32, copy=True), lora, key)
                    merged_t = w.to(dtype=orig.dtype).cpu()
                    del w
                    n_lora += 1
                if float_q:
                    w = merged_t.to(device=device, dtype=torch.float32)
                    if out_low and _low_fits(tuple(w.shape)):
                        _set_quantized(sd, key, *_quantize_low(w, device, fmt))
                        n_low += 1
                    else:
                        qdata, scale, qconf = _quantize_int8(w, device)
                        _set_quantized(sd, key, qdata, {"weight_scale": scale}, qconf)
                    del w
                    n_newq += 1
                else:
                    sd[key] = merged_t

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged = sum(1 for p in plan if p[6])
        kept = len([s for s in a_short if s not in companions]) - merged
        msg = (f"{LOG} passe unique en {time.time() - t0:.1f}s : {kept} tenseurs gardes de unet_1, "
               f"{n_copy_b} copies exactes de unet_2, {n_requant} fusionnes + requantifies INT8, "
               f"{n_float} fusionnes en float")
        if n_newq:
            msg += f", {n_newq} matrices bf16/fp32 quantifiees"
        if n_bf16:
            msg += f", {n_bf16} couches INT8 remises en bf16"
        if out_low:
            msg += f", {n_low} couches converties en {LOW_NAMES[fmt]}"
            if n_low_skip:
                msg += (f" ({n_low_skip} restees en INT8 : in_features non multiple de "
                        f"{LOW_CONVROT_GROUPSIZE})")
        if n_lora:
            msg += f", LoRA integres dans {n_lora} tenseurs"
        if only_a:
            msg += f", {only_a} propres a unet_1 (gardes"
            msg += f" : {', '.join(extras_kept[:4])})" if extras_kept else ")"
        if dropped:
            msg += f", {', '.join(dropped)} retires (repere adaLN de unet_2)"
        if only_b:
            msg += f", {only_b} propres a unet_2 (ignores)"
        print(msg + f". [{_ram()}]", flush=True)
        return {"kept": kept, "copied": n_copy_b, "requantized": n_requant, "float": n_float,
                "lora_tensors": n_lora, "lowbit": n_low, "lowbit_skipped": n_low_skip,
                "float_quantized": n_newq, "to_bf16": n_bf16,
                "adaln_table_diff": table_diff,
                "adaln_from": None if adaln_lock is None else ("unet_2" if adaln_lock == 0.0 else "unet_1")}

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save_sd(self, sd, metadata, filename_prefix):
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        out_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")

        if not sd:
            raise RuntimeError("state_dict vide : rien a sauvegarder")
        if len(sd) < 100:
            raise RuntimeError(f"state_dict suspect ({len(sd)} tenseurs) : sauvegarde annulee")

        meta = {str(a): str(b) for a, b in (metadata or {}).items()}
        _write_safetensors(sd, out_path, meta)
        size = os.path.getsize(out_path)
        if size < 1024 * 1024:
            raise RuntimeError(f"fichier ecrit anormalement petit ({size} octets) : {out_path}")
        print(f"{LOG} sauvegarde ({size / (1024 ** 3):.2f} Go, {len(sd)} tenseurs) -> {out_path} "
              f"[{_ram()}]", flush=True)

    # ------------------------------------------------------------------
    # Entree
    # ------------------------------------------------------------------

    def process(self, unet_1, unet_2, ratio_global, ratio_in_out, ratio_text,
                ratio_blocks_early, ratio_blocks_mid, ratio_blocks_late,
                early_end, mid_end, ratio_adaln, adaln_start, adaln_end,
                save, filename_prefix,
                lora_1="None", strength_1=1.0,
                lora_2="None", strength_2=1.0,
                lora_3="None", strength_3=1.0, output_format="int8"):
        low = output_format.replace(" HQ", "")
        if low in LOW_FORMATS:
            ok, reason = _low_available(low)
            if not ok:
                raise RuntimeError(f"Sortie {LOW_NAMES[low]} indisponible : {reason}. "
                                   "Rien n'a ete fait.")
        if early_end > mid_end:
            raise ValueError(f"early_end ({early_end}) doit etre <= mid_end ({mid_end})")
        if adaln_start > adaln_end:
            raise ValueError(f"adaln_start ({adaln_start}) doit etre <= adaln_end ({adaln_end})")
        cfg = {
            "global": ratio_global, "in_out": ratio_in_out, "text": ratio_text,
            "early": ratio_blocks_early, "mid": ratio_blocks_mid, "late": ratio_blocks_late,
            "early_end": early_end, "mid_end": mid_end,
            "adaln": ratio_adaln, "adaln_start": adaln_start, "adaln_end": adaln_end,
        }

        print(f"{LOG} --- debut (save={save}, sortie {output_format}) ---", flush=True)
        freed = _free_unused_cache()
        print(f"{LOG} cache libere avant le merge : {freed / 1024 ** 3:.1f} Go [{_ram()}]", flush=True)
        device = comfy.model_management.get_torch_device()

        path1 = folder_paths.get_full_path("diffusion_models", unet_1)
        no_merge = unet_2 in (None, "", "None")
        path2 = None if no_merge else folder_paths.get_full_path("diffusion_models", unet_2)
        if path1 is None or (path2 is None and not no_merge):
            raise ValueError(f"modele introuvable : {unet_1 if path1 is None else unet_2}")
        if not no_merge and not path2.lower().endswith((".safetensors", ".sft")):
            raise ValueError("unet_2 doit etre un fichier .safetensors (lecture tenseur par tenseur).")

        # 1. unet_1 complet : c'est lui qui devient le modele fusionne.
        if path1.lower().endswith((".safetensors", ".sft")):
            # Lecture sans mmap : chaque tenseur remplace par le merge libere
            # aussitot sa RAM (avec un mmap, le fichier entier y resterait).
            with _SafetensorsFile(path1) as f1:
                metadata = f1.metadata()
                sd = {k: f1.get_tensor(k) for k in f1.keys()}
        else:
            sd, metadata = comfy.utils.load_torch_file(path1, return_metadata=True)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        # Config de quantification en metadonnees (ancien format) -> cles comfy_quant.
        sd, metadata = comfy.utils.convert_old_quants(sd, prefix, metadata=dict(metadata or {}))
        metadata = {k: v for k, v in (metadata or {}).items() if k != "_quantization_metadata"}
        print(f"{LOG} charge : {unet_1} ({len(sd)} tenseurs) [{_ram()}]", flush=True)

        # LoRA charges avant la passe : une erreur arrete tout sans rien modifier.
        specs = [(lora_1, strength_1), (lora_2, strength_2), (lora_3, strength_3)]
        lora_targets, lora_infos = _load_loras(sd, prefix, metadata, specs)
        # Sans sauvegarde, le LoRA n'est PAS integre aux poids : il est applique
        # au MODEL par le chargeur de ComfyUI, exactement comme un Load LoRA
        # branche apres le node. Il n'est integre que pour ecrire le fichier.
        bake = save == "enabled" and bool(lora_targets)
        if not bake:
            lora_targets = None

        # 2. unet_2 en lecture paresseuse, 3. passe unique.
        if no_merge:
            # Pas de merge : ratios forces a 1.0 (tout vient de unet_1).
            cfg = dict(cfg, **{"global": 1.0, "in_out": -1.0, "text": -1.0, "early": -1.0,
                               "mid": -1.0, "late": -1.0, "adaln": -1.0})
            print(f"{LOG} unet_2 = None : pas de merge, LoRA et format de sortie seulement.",
                  flush=True)
            stats = self._merge(sd, prefix, _SelfSource(sd, prefix), cfg, device, lora_targets,
                                out_format=output_format)
        else:
            with _SafetensorsFile(path2) as handle:
                src = _LazySource(handle)
                print(f"{LOG} ouvert : {unet_2} ({len(src.keys)} tenseurs)", flush=True)
                stats = self._merge(sd, prefix, src, cfg, device, lora_targets,
                                    out_format=output_format)
        del lora_targets
        gc.collect()

        metadata["h3_merge"] = json.dumps({
            "unet_1": unet_1, "unet_2": unet_2, "ratios": cfg,
            "loras": lora_infos if bake else [],
            "output_format": output_format,
            "stats": stats,
        })

        # 4. Sauvegarde AVANT construction : load_diffusion_model_state_dict()
        #    consomme le state_dict.
        if save == "enabled":
            self._save_sd(sd, metadata, filename_prefix)

        # 5. Construction : ComfyUI lit les cles comfy_quant et active l'INT8 natif.
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={}, metadata=metadata)
        if model is None:
            raise ValueError("construction du modele impossible")
        del sd

        # 6. LoRA en direct : meme code que LoraLoaderModelOnly.
        if lora_infos and not bake:
            for info in lora_infos:
                lora = comfy.utils.load_torch_file(
                    folder_paths.get_full_path("loras", info["name"]), safe_load=True)
                model, _ = comfy.sd.load_lora_for_models(model, None, lora, info["strength"], 0)
                del lora
                print(f"{LOG} LoRA {info['name']} (force {info['strength']:g}) applique par "
                      "ComfyUI, comme un Load LoRA (non integre aux poids : save=disable).",
                      flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"{LOG} --- fin --- [{_ram()}]", flush=True)
        return (model,)
