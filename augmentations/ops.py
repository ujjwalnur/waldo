import math
from dataclasses import dataclass
from typing import Tuple, List


from omegaconf import MISSING
import tensorflow as tf
from .base import ClassificationAugmentationBase


@dataclass
class AugmentationList:
    transforms: List[ClassificationAugmentationBase] = MISSING

    def transform(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        out = (image, label)
        for tx in self.transforms:
            out = tx.transform(out[0], out[1])
        return out


@dataclass
class ResizeExplicit(ClassificationAugmentationBase):
    new_height: int
    new_width: int

    def map(self, image, label) -> Tuple[tf.Tensor, tf.Tensor]:
        image_out = tf.image.resize(image, size=(self.new_height, self.new_width))
        image_out = tf.cast(image_out, tf.uint8)
        return image_out, label


@dataclass
class ResizeSmallest(ClassificationAugmentationBase):
    resize_dim: int

    def map(self, image, label) -> Tuple[tf.Tensor, tf.Tensor]:
        image_shape = tf.shape(image)
        ar = image_shape[0] / image_shape[1]
        image_out = tf.cond(
            image_shape[0] < image_shape[1],
            lambda: self._height_is_smallest(image, ar),
            lambda: self._width_is_smallest(image, ar),
        )
        image_out = tf.cast(image_out, tf.uint8)
        return image_out, label

    def _height_is_smallest(self, image, aspect_ratio):
        w_new = self.resize_dim / aspect_ratio
        w_new = tf.cast(w_new, tf.int32)
        return tf.image.resize(image, (self.resize_dim, w_new))

    def _width_is_smallest(self, image, aspect_ratio):
        h_new = self.resize_dim * aspect_ratio
        h_new = tf.cast(h_new, tf.int32)
        return tf.image.resize(image, (h_new, self.resize_dim))


@dataclass
class ResizeLargest(ClassificationAugmentationBase):
    resize_dim: int

    def map(self, image, label):
        image_shape = tf.shape(image)
        ar = image_shape[0] / image_shape[1]
        image_out = tf.cond(
            image_shape[0] < image_shape[1],
            lambda: self._width_is_largest(image, ar),
            lambda: self._height_is_largest(image, ar),
        )
        image_out = tf.cast(image_out, tf.uint8)
        return image_out, label

    def _height_is_largest(self, image, aspect_ratio):
        w_new = self.resize_dim / aspect_ratio
        w_new = tf.cast(w_new, tf.int32)
        return tf.image.resize(image, (self.resize_dim, w_new))

    def _width_is_largest(self, image, aspect_ratio):
        h_new = self.resize_dim * aspect_ratio
        h_new = tf.cast(h_new, tf.int32)
        return tf.image.resize(image, (h_new, self.resize_dim))


@dataclass
class HorizontalFlip(ClassificationAugmentationBase):
    def map(self, image, label):
        return tf.image.flip_left_right(image), label


@dataclass
class RandomCrop(ClassificationAugmentationBase):
    crop_height: int
    crop_width: int

    def map(self, image, label):
        image_out = tf.image.random_crop(image, (self.crop_height, self.crop_width, 3))
        return image_out, label


@dataclass
class NormalizeImageRange(ClassificationAugmentationBase):
    init_minval: int = 0
    init_maxval: int = 255
    target_minval: float = -1.0
    target_maxval: float = 1.0

    def map(self, image, label):
        image = tf.cast(image, tf.float32)
        image -= float(self.init_minval)
        image *= (self.target_maxval - self.target_minval) / (
            float(self.init_maxval) - float(self.init_minval)
        )
        image += self.target_minval
        return image, label

    def skip(self, image, label):
        return tf.cast(image, tf.float32), label


@dataclass
class BrightnessContrast(ClassificationAugmentationBase):
    min_delta: float = -0.3
    max_delta: float = 0.3
    contrast_factor_min: float = 0.4
    contrast_factor_max: float = 1.5

    def map(self, image, label):
        image = tf.cast(image, tf.float32)
        delta_value = tf.random.uniform(
            shape=(), minval=self.min_delta + 1e-04, maxval=self.max_delta
        )
        image = tf.image.adjust_brightness(image, delta=delta_value)
        contrast_factor = tf.random.uniform(
            shape=(), minval=self.contrast_factor_min, maxval=self.contrast_factor_max
        )
        image = tf.image.adjust_contrast(image, contrast_factor=contrast_factor)
        image = tf.cast(image, tf.uint8)
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
        return image, label


@dataclass
class Grayscale(ClassificationAugmentationBase):
    def map(self, image, label):
        image_out = tf.image.rgb_to_grayscale(image)
        image_out = tf.tile(image_out, (1, 1, 3))
        return image_out, label


@dataclass
class CenterCrop(ClassificationAugmentationBase):
    crop_height: int
    crop_width: int

    def map(self, image, label):
        image_shape = tf.shape(image)
        left = tf.cast(tf.math.ceil((image_shape[1] - self.crop_width) / 2), tf.int32)
        top = tf.cast(tf.math.ceil((image_shape[0] - self.crop_height) / 2), tf.int32)
        center_cropped = tf.slice(
            image, begin=(top, left, 0), size=(self.crop_height, self.crop_width, 3)
        )
        return center_cropped, label


@dataclass
class RandomRotate(ClassificationAugmentationBase):
    min_rot: int
    max_rot: int
    fill_mode: str

    def __post_init__(self):
        super(RandomRotate, self).__post_init__()
        self.layer = tf.keras.layers.RandomRotation(
            factor=(self.min_rot / 360.0, self.max_rot / 360.0),
            fill_mode=self.fill_mode,
        )

    def map(self, image, label):
        rotated_image = self.layer(image)
        rotated_image = tf.cast(rotated_image, tf.uint8)
        return rotated_image, label


@dataclass
class RandomTranslation(ClassificationAugmentationBase):
    height_factor_min: float
    height_factor_max: float
    width_factor_min: float
    width_factor_max: float
    fill_mode: str

    def __post_init__(self):
        super(RandomTranslation, self).__post_init__()
        self.layer = tf.keras.layers.RandomTranslation(
            height_factor=(self.height_factor_min, self.height_factor_max),
            width_factor=(self.width_factor_min, self.width_factor_max),
            fill_mode=self.fill_mode,
        )

    def map(self, image, label):
        translated_image = self.layer(image)
        translated_image = tf.cast(translated_image, tf.uint8)
        return translated_image, label
