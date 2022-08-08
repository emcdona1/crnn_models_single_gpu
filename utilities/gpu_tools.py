from typing import Union
import tensorflow as tf
import subprocess
import numpy as np

def gpu_selection() -> Union[int, bool]:
    available_gpus = tf.config.list_physical_devices('GPU')
    if available_gpus:
        command = 'nvidia-smi --query-gpu=memory.free --format=csv'
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return int(np.argmax(memory_free_values))
    else:
        return False
