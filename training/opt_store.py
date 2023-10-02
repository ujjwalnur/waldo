import tensorflow as tf
from omegaconf import OmegaConf, DictConfig, open_dict
from .lr_store import get_lr_schedule

OPTIMIZER_STORE = {
    "adafactor": tf.keras.optimizers.Adafactor,
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.experimental.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "nadam": tf.keras.optimizers.experimental.Nadam,
    "ftrl": tf.keras.optimizers.experimental.Ftrl,
    "adamax": tf.keras.optimizers.experimental.Adamax,
    "adagrad": tf.keras.optimizers.experimental.Adagrad,
    "adadelta": tf.keras.optimizers.experimental.Adadelta,
    "adamw": tf.keras.optimizers.AdamW,
    "lion": tf.keras.optimizers.Lion,
}


def build_optimizer(cfg: DictConfig) -> tf.keras.optimizers.Optimizer:
    with open_dict(cfg):
        lr_schedule_config = cfg.pop("lr_schedule")
        optimizer_name = cfg.pop("name")
        lr_schedule = get_lr_schedule(lr_schedule_config)
        # cfg["learning_rate"] = get_lr_schedule(lr_schedule_config)

    if optimizer_name not in OPTIMIZER_STORE:
        raise KeyError(f"Optimizer {optimizer_name} is not a supported optimizer.")

    return OPTIMIZER_STORE[optimizer_name](learning_rate=lr_schedule, **cfg)
