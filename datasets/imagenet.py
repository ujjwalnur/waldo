import os
from typing import Literal, Union
import tensorflow_datasets as tfds
from augmentations.ops import AugmentationList
import tensorflow as tf


def prepare_imagenet_tfrecords(
    imagenet_download_dir: Union[str, os.PathLike],
    tfrecord_dir: Union[str, os.PathLike],
):
    builder = tfds.builder(name="imagenet2012", data_dir=tfrecord_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=imagenet_download_dir)
    )
    return None


def get_imagenet_tfrecords(
    tfrecord_dir: Union[str, os.PathLike],
    split: Literal["train", "validation"],
    transforms: AugmentationList,
    batch_size: int,
):
    builder = tfds.builder(name="imagenet2012", data_dir=tfrecord_dir)

    ds = builder.as_dataset(split=split)
    if split == "train":
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    ds = ds.map(lambda x: map_fn(x, transforms), num_parallel_calls=50)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=100)
    return ds


def map_fn(x, augmentation_list: AugmentationList):
    image = x["image"]
    label = x["label"]
    label = tf.one_hot(label, depth=1000)
    out = augmentation_list.transform(image, label)
    return out
