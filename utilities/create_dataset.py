import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import tensorflow as tf
from tensorflow import keras
from .configurations import TunerConfiguration, TrainerConfiguration, TestConfiguration
from abc import ABC, abstractmethod


class HandwritingDataset(ABC):
    def __init__(self):
        self.c = None
        self.folder = None
        self.metadata_filename = None
        self.metadata = None
        self.CHAR_LIST: str = '\' !"#&()[]*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.characters = sorted(set(list(self.CHAR_LIST)))

        # Mapping characters to integers
        self.char_to_num = keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
    
    @abstractmethod
    def create_dataset(self, batch_size: int, image_folder: Path = '', metadata_filename=''):
        pass
    
    def _load_images(self):
        self.metadata = pd.read_csv(self.metadata_filename)
        col = self.metadata[self.c.metadata_image_column]
        self.metadata['word_image_basenames'] = col.apply(lambda f: os.path.basename(Path(f)))

        images = list()
        images.extend(self.folder.rglob('*.png'))
        images.extend(self.folder.rglob('*.jpg'))
        images = sorted(list(map(str, images)))
        print(len(images))

        labels = [os.path.basename(l) for l in images]
        labels = [self.metadata[self.metadata['word_image_basenames'] == b] for b in labels]
        labels = [b[self.c.metadata_transcription_column].item() for b in labels]
        labels = [str(e).ljust(self.c.max_label_length) for e in labels]
        
        return images, labels
    
    def _encode_dataset(self, batch_size, images, labels) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = (
            dataset.map(
                self._encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return dataset
    
    def _encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        if '.png' in str(img_path):
            img = tf.io.decode_png(img, channels=1)
        else:
            img = tf.io.decode_jpeg(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [self.c.img_height, self.c.img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        # 7. Return a dict as our model is expecting two inputs
        return {'image': img, 'label': label}


class TrainDataset(HandwritingDataset):
    def __init__(self, configuration: Union[TrainerConfiguration, TunerConfiguration]):
        super().__init__()
        self.c = configuration
        self.train_dataset = None
        self.validation_dataset = None
        self.images_train = None
        self.images_valid = None
        self.labels_train = None
        self.labels_valid = None
        self.train_size = 0
        self.validation_size = 0

    def _split_data(self, images, labels, train_size=0.9, shuffle=True):
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

    def create_dataset(self, batch_size: int, image_folder: Path = '', metadata_filename=''):
        self.folder = (image_folder.absolute() if image_folder else self.c.image_set_location)
        self.metadata_filename = Path(self.folder, metadata_filename).absolute() \
            if metadata_filename else self.c.metadata_file_name
        images, labels = super(TrainDataset, self)._load_images()
        self.images_train, self.images_valid, self.labels_train, self.labels_valid = self._split_data(np.array(images),
                                                                                                      np.array(labels))

        print(f'Training images ({self.images_train.shape[0]}) and labels ({self.labels_train.shape[0]}) loaded.')
        print(f'Validation images ({self.images_valid.shape[0]}) and labels ({self.labels_valid.shape[0]}) loaded.')
        self.train_size = self.images_train.shape[0]
        self.validation_size = self.images_valid.shape[0]

        self.train_dataset = self._encode_dataset(batch_size, self.images_train, self.labels_train)
        self.validation_dataset = self._encode_dataset(batch_size, self.images_valid, self.labels_valid)

    def update_batch_size(self, batch_size: int):
        self.train_dataset = self._encode_dataset(batch_size, self.images_train, self.labels_train)
        self.validation_dataset = self._encode_dataset(batch_size, self.images_valid, self.labels_valid)


class TestDataset(HandwritingDataset):
    def __init__(self, configuration: Union[Path, str, TestConfiguration]):
        super().__init__()
        self.c = configuration \
            if isinstance(configuration, TestConfiguration) \
            else TestConfiguration(configuration)
        self.test_dataset = None
        self.size = None    
    
    def create_dataset(self, batch_size: int, image_folder: Path = '', metadata_filename=''):
        self.folder = (image_folder.absolute() if image_folder else self.c.image_set_location)
        self.metadata_filename = Path(self.folder, metadata_filename).absolute() \
            if metadata_filename else self.c.metadata_file_name
        images, labels = super(TestDataset, self)._load_images()

        self.size = len(images)
        x_test = np.array(images)
        y_test = np.array(labels)
        self.test_dataset = self._encode_dataset(batch_size, x_test, y_test)

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :self.c.max_label_length
                  ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode('utf-8')
            output_text.append(res)
        return output_text
