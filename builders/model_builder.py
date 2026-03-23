import torch
import torch.nn as nn
from collections import OrderedDict
from models.U_Net import U_Net
from models.UNeXt import UNext
from models.SegNet import SegNet
from models.deepcrack import DeepCrack
from models.SSNet import SSNet_T, SSNet_S, SSNet_M
from models.LECSFormer import LECSFormer
from models.hybrid_segmentor import HybridSegmentor

# A model registry that maps model names to their respective classes
MODEL_REGISTRY = {
    "U_Net": U_Net,  "UNeXt": UNext, "SegNet": SegNet,
     "deepcrack": DeepCrack,
     "SSNet_T": SSNet_T, "SSNet_S": SSNet_S, "SSNet_M": SSNet_M,
    "LECSFormer": LECSFormer, "HybridSegmentor": HybridSegmentor}


def get_model(args):
    """
    Initializes a model from the registry based on the provided arguments.
    """
    model_class = MODEL_REGISTRY.get(args.model)
    if not model_class:
        raise ValueError(f"Model {args.model} not recognized. Available models: {list(MODEL_REGISTRY.keys())}")

    # Pass arguments to the model's constructor
    if args.model == "LECSFormer":
        return model_class(img_size=[256, 256], in_channels=3, num_classes=args.num_classes)
    elif args.model == "HybridSegmentor":
        return model_class(in_channels=3, num_classes=args.num_classes)
    elif args.model == "UNeXt":
        return model_class(in_channels=3, num_classes=args.num_classes)
    elif args.model == "SSNet_T":
        return model_class(in_channels=3, num_classes=args.num_classes)
    else:
        return model_class(in_channels=3, num_classes=args.num_classes)


def load_weight(model_path, model):
    """
    Loads model weights from a checkpoint file.
    This function is robust to checkpoints saved from both single-GPU and multi-GPU (nn.DataParallel) training.
    """
    checkpoint = torch.load(model_path, map_location='cpu')

    # The state dictionary is nested under 'model_state_dict' in our training script
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Create a new state dictionary to hold the cleaned keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # If the key starts with 'module.', it's from a DataParallel checkpoint.
        # We strip this prefix to match the keys of a non-DataParallel model.
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load the cleaned state dictionary
    model.load_state_dict(new_state_dict)

    return model

