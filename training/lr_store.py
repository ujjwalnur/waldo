import tensorflow as tf
from omegaconf import OmegaConf, DictConfig, open_dict


def get_lr_schedule(cfg: DictConfig):
    with open_dict(cfg):
        schedule_name = cfg.pop("name", None)

    if schedule_name == "constant":
        return cfg["lr"]
    elif schedule_name == "cosine_decay":
        return tf.keras.optimizers.schedules.CosineDecay(**cfg)
    elif schedule_name == "cosine_decay_restarts":
        return tf.keras.optimizers.schedules.CosineDecayRestarts(**cfg)
    elif schedule_name == "exponential_decay":
        return tf.keras.optimizers.schedules.ExponentialDecay(**cfg)
    elif schedule_name == "inverse_time_decay":
        return tf.keras.optimizers.schedules.InverseTimeDecay(**cfg)
    elif schedule_name == "piecewise_constant":
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(**cfg)
    elif schedule_name == "polynomial_decay":
        return tf.keras.optimizers.schedules.PolynomialDecay(**cfg)

    else:
        raise NameError(f"Invalid schedule: {schedule_name}")
