from typing import Literal

from .imagenet import get_imagenet_tfrecords
from omegaconf import DictConfig, open_dict, OmegaConf
from augmentations.ops import AugmentationList


def get_dataset(
    cfg: DictConfig, split: Literal["train", "validation"], transforms: AugmentationList
):
    # Here we convert the omegaconf to python dictionary.
    # Currently it is not allowed to push non-primitive values into an OmegaConf DictConfig.
    # So, this is a workaround.
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset_name = cfg.pop("name")
    cfg["transforms"] = transforms
    if dataset_name == "imagenet":
        dataset = get_imagenet_tfrecords(split=split, **cfg)
        return dataset
