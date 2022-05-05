from pathlib import Path
from tensorflow.keras import layers


class TrainerConfiguration:
    def __init__(self):
        self.NUM_EPOCHS = 15
        self.SEED = 2
        self.EARLY_STOPPING = True
        # Desired image dimensions
        self.img_width = 400
        self.img_height = 100
        self.IMAGE_FORMAT = 'png'
        self.PROJECT_FOLDER = ''
        self.IMAGE_SET_NAME = 'words'
        self.data_dir = Path(self.PROJECT_FOLDER, self.IMAGE_SET_NAME)
        self.metadata_file_name = 'word_metadata2.csv'
        
        self.CHAR_LIST: str = '\' !"#&()[]*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.MAX_LABEL_LENGTH = 30
        self.characters = sorted(set(list(self.CHAR_LIST)))
        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None, invert=True
        )