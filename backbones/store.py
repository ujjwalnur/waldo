from typing import Tuple

from omegaconf import OmegaConf, DictConfig, open_dict
from .resnet import get_resnet
from hydra.utils import instantiate


def build_backbone(
    cfg: DictConfig, input_shape: Tuple[int, int, int], num_classes: int
):
    with open_dict(cfg):
        backbone_name = cfg.pop("name", None)
        cfg["input_shape"] = input_shape
        cfg["num_classes"] = num_classes
    if "resnet_" in backbone_name:
        resnet = get_resnet(**cfg)
        return resnet
    return None
