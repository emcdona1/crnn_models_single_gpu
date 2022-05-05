import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path, PureWindowsPath
import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers
from trainer.trainer_config import TrainerConfiguration


def create_dataset(batch_size: int):
    c = TrainerConfiguration()
    header = None
    line_list = list()
    with open(Path(c.data_dir, c.metadata_file_name), mode='r', encoding='utf-8') as f:
        for line in f:
            if not header:
                header = line.replace('\n', '').split(',')
            else:
                line = line.replace('\n', '').split(',', maxsplit=2)
                line_list.append(line)
    metadata = pd.DataFrame(line_list, columns=header)

    images = sorted(list(map(str, list(c.data_dir.rglob(f'*.{c.IMAGE_FORMAT}')))))
    print(len(images))
    labels = list()
    metadata['word_image_basenames'] = metadata['image_location'].apply(lambda f: os.path.basename(PureWindowsPath(f).as_posix()))

    labels = [os.path.basename(l) for l in images]
    labels = [metadata[metadata['word_image_basenames'] == b] for b in labels]
    labels = [b['transcription'].item() for b in labels]
    labels = [str(e).ljust(c.MAX_LABEL_LENGTH) for e in labels]

    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

    print(f'Training images ({x_train.shape[0]}) and labels ({y_train.shape[0]}) loaded.')
    print(f'Validation images ({x_valid.shape[0]}) and labels ({y_valid.shape[0]}) loaded.')

    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolution blocks and each block will have
    # a pooling layer which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    downsample_factor = 4
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return train_dataset, validation_dataset


def encode_single_sample(img_path, label):
    c = TrainerConfiguration()
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    if c.IMAGE_FORMAT == 'png':
        img = tf.io.decode_png(img, channels=1)
    else:
        img = tf.io.decode_jpeg(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [c.img_height, c.img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = c.char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    # 7. Return a dict as our model is expecting two inputs
    return {'image': img, 'label': label}


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
