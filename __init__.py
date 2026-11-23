"""MinimaxH3 TotalMerger - pack autonome pour ComfyUI.

Merge de deux modeles MiniMax H3 par groupes de blocs, avec des entrees INT8
ConvRot (format natif ComfyUI) ou bf16. Les poids INT8 sont dequantifies,
fusionnes puis requantifies au format du modele de base ; un ratio de 1.0 ou
0.0 recopie le tenseur source a l'identique.

Entierement natif : dequantification, quantification, forward et
deserialisation sont assures par comfy.quant_ops / comfy_kitchen.
"""

import logging

try:
    from .merge_node import ModelMergeMiniMaxH3Int8

    NODE_CLASS_MAPPINGS = {
        "ModelMergeMiniMaxH3Int8": ModelMergeMiniMaxH3Int8,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelMergeMiniMaxH3Int8": "MinimaxH3 TotalMerger",
    }
except Exception as e:
    logging.error(f"H3 Merge: import du node impossible: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
