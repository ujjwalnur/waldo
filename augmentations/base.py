import abc
from dataclasses import dataclass
from typing import Tuple, List
import tensorflow as tf
import tensorflow_probability as tfp
from omegaconf import MISSING


@dataclass
class ClassificationAugmentationBase(abc.ABC):
    p: float

    def __post_init__(self):
        self._dist: tfp.distributions.Distribution = tfp.distributions.Bernoulli(
            probs=self.p, dtype=tf.bool
        )
        return None

    @abc.abstractmethod
    def map(self, image, label):
        pass

    def skip(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return image, label

    def transform(self, image, label):
        out = tf.cond(
            self._dist.sample(),
            lambda: self.map(image, label),
            lambda: self.skip(image, label),
        )
        return out
