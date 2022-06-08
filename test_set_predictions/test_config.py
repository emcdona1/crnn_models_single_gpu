import configparser
from pathlib import Path


class TestConfiguration:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path('..', 'setup.cfg'))
        self.IMAGE_SET_NAME = config['test']['TEST_IMAGE_SET_NAME']
        self.METADATA_FILE_NAME = config['test']['TEST_METADATA_FILE_NAME']
        self.IMAGE_FORMAT = config['project']['IMAGE_FORMAT']
        self.MAX_LABEL_LENGTH = config.getint('project', 'MAX_LABEL_LENGTH')
        self.BATCH_SIZE =  config.getint('project', 'BATCH_SIZE')
        self.IMG_HEIGHT =  config.getint('project', 'IMG_HEIGHT')
        self.IMG_WIDTH =  config.getint('project', 'IMG_WIDTH')