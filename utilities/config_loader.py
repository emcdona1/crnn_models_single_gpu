from abc import ABC
import configparser
from pathlib import Path
from tensorflow.keras import layers


class HandwritingConfiguration(ABC):
    def __init__(self):
        self.config = configparser.ConfigParser()
        if not self.config.read('./setup.cfg'):
            print('WARNING: No configuration file loaded.')
            exit(1)
        self.IMAGE_SET_LOCATION = Path(self.config['test']['TEST_IMAGE_SET_NAME'])
        self.METADATA_FILE_NAME = self.config['test']['TEST_METADATA_FILE_NAME']
        self.IMAGE_FORMAT = self.config['project']['IMAGE_FORMAT']
        self.MAX_LABEL_LENGTH = self.config.getint('project', 'MAX_LABEL_LENGTH')
        self.BATCH_SIZE = self.config.getint('project', 'BATCH_SIZE')
        self.IMG_HEIGHT = self.config.getint('project', 'IMG_HEIGHT')
        self.IMG_WIDTH = self.config.getint('project', 'IMG_WIDTH')
        self._char_list: str = '\' !"#&()[]*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.MAX_LABEL_LENGTH = 30

        self._characters = sorted(set(list(self._char_list)))
        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self._characters), mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self._characters), mask_token=None, invert=True
        )


class TrainerConfiguration(HandwritingConfiguration):
    def __init__(self):
        super().__init__()
        self.NUM_EPOCHS = self.config.getint('train', 'NUM_EPOCHS')
        self.SEED = self.config.getint('train', 'SEED')
        self.EARLY_STOPPING = self.config.getboolean('train', 'EARLY_STOPPING')


class TestConfiguration(HandwritingConfiguration):
    def __init__(self):
        super().__init__()
