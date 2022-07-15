from abc import ABC
import configparser
from pathlib import Path
from tensorflow.keras import layers


class HandwritingConfiguration(ABC):
    def __init__(self):
        self.config = configparser.ConfigParser()
        if not self.config.read('../setup.cfg'):
            print('WARNING: No configuration file loaded.')
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

        self.NUM_EPOCHS = 75
        self.SEED = 2
        self.EARLY_STOPPING = True
        # Desired image dimensions
        self.PROJECT_FOLDER = ''  # only use if the Dockerfile is not in the same file directory level as IMAGE_SET_FOLDER
        self.IMAGE_SET_FOLDER = 'IAM_Words'
        self.data_dir = Path(self.PROJECT_FOLDER, self.IMAGE_SET_FOLDER)


class TestConfiguration(HandwritingConfiguration):
    def __init__(self):
        super().__init__()
