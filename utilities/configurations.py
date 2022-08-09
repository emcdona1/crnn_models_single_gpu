from abc import ABC
import configparser
from pathlib import Path
from tensorflow import keras
from typing import Union


class HandwritingConfiguration(ABC):
    def __init__(self, config_file_location: Union[Path, str]):
        self.config = configparser.ConfigParser()
        if not self.config.read(Path(config_file_location)):
            raise FileNotFoundError(f'{config_file_location} does not exist or is not a configuration file.')
        self.base_directory = config_file_location.absolute().parent
        self.image_format = self.config['project']['IMAGE_FORMAT']
        self.max_label_length = self.config.getint('project', 'MAX_LABEL_LENGTH')
        self.batch_size = self.config.getint('project', 'BATCH_SIZE')
        self.img_height = self.config.getint('project', 'IMG_HEIGHT')
        self.img_width = self.config.getint('project', 'IMG_WIDTH')
        self.metadata_file_name = self.config['project']['METADATA_FILE_NAME']
        self.metadata_image_column = self.config['project']['METADATA_IMAGE_COLUMN']
        self.metadata_transcription_column = self.config['project']['METADATA_TRANSCRIPTION_COLUMN']
        self._char_list: str = '\' !"#&()[]*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        self._characters = sorted(set(list(self._char_list)))
        # Mapping characters to integers
        self.char_to_num = keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self._characters), mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self._characters), mask_token=None, invert=True
        )


class TrainerConfiguration(HandwritingConfiguration):
    def __init__(self, config_file_location: Union[Path, str]):
        super().__init__(config_file_location)
        self.image_set_location = Path(self.config['train']['IMAGE_SET_NAME'])
        if not self.image_set_location.is_absolute():
            self.image_set_location = Path(self.base_directory, self.image_set_location)
        self.metadata_file_name = Path(self.image_set_location, self.metadata_file_name)
        self.num_epochs = self.config.getint('train', 'NUM_EPOCHS')
        self.seed = self.config.getint('train', 'SEED')
        self.early_stopping = self.config.getboolean('train', 'EARLY_STOPPING')
        self.learning_rate = self.config.getfloat('train', 'LEARNING_RATE')


class TestConfiguration(HandwritingConfiguration):
    def __init__(self, config_file_location: Union[Path, str]):
        super().__init__(config_file_location)
        self.image_set_location = Path(self.config['test']['IMAGE_SET_NAME'])
        if not self.image_set_location.is_absolute():
            self.image_set_location = Path(self.base_directory, self.image_set_location)
        self.metadata_file_name = Path(self.image_set_location, self.metadata_file_name)