import tensorflow as tf
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
from backbones.store import build_backbone
from training.opt_store import build_optimizer
import augmentations.ops
from training.trainer import build_trainer_from_config
from pprint import PrettyPrinter
import os

pp = PrettyPrinter(indent=4)


@hydra.main(
    version_base=None, config_path="waldo_conf", config_name="resnet50_imagenet"
)
def my_app(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    trainer = build_trainer_from_config(cfg)
    trainer.run()
    return None


#
# @hydra.main(version_base=None, config_path=".", config_name="abc")
# def my_app(cfg: DictConfig):
#     preprocessing_train = cfg.preprocessing_training
#     obj = instantiate(preprocessing_train)
#     print(obj)


if __name__ == "__main__":
    my_app()
