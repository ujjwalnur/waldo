import math
import tensorflow as tf
import numpy as np
import cv2
import random


def make_image_grid(
    batched_image_array,
    num_to_plot=16,
    space=20,
    image_size=(256, 256),
    min_val=-1.0,
    max_val=1.0,
):
    num_images = batched_image_array.shape[0]
    if num_to_plot < num_images:
        idx = random.sample(range(num_images), num_to_plot)
        batched_image_array = batched_image_array[idx, :, :, :]
    if tf.keras.backend.image_data_format() == "channels_first":
        batched_image_array = np.transpose(batched_image_array, (0, 2, 3, 1))

    num_row = int(math.sqrt(num_to_plot))
    grid_image = np.zeros(
        shape=(
            image_size[0] * num_row + space * (num_row - 1),
            image_size[1] * num_row + space * (num_row - 1),
            3,
        ),
        dtype=np.uint8,
    )
    for batch_num in range(batched_image_array.shape[0]):
        row_index = batch_num % num_row
        col_index = batch_num // num_row
        img = batched_image_array[batch_num, :, :, :]
        img = cv2.resize(img, (image_size[1], image_size[0]))
        img -= min_val
        img *= 255.0
        img /= max_val - min_val
        img = img.astype(np.uint8)
        row_init = row_index * (image_size[0] + space)
        col_init = col_index * (image_size[1] + space)
        grid_image[
            row_init : row_init + image_size[0], col_init : col_init + image_size[1], :
        ] = img
    return grid_image
