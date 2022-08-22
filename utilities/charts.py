import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


class LossChart:
    def __init__(self):
        self.folder_name = Path('graphs')
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.timestamp: str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
        base_filename: str = f'loss-{self.timestamp}.png'
        self.path: Path = Path(self.folder_name, base_filename)

    def create_chart(self, history: keras.callbacks.History) -> None:
        plt.figure(2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(str(self.path))
        plt.clf()
