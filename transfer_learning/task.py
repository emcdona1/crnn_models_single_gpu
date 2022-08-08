import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress oneDNN INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from utilities import TrainDataset, TrainerConfiguration
from utilities import gpu_selection, CTCLayer


def main():
    base_model = keras.models.load_model(base_model_path, custom_objects={'CTCLayer': CTCLayer})
    data = TrainDataset(c)
    data.create_dataset(c.batch_size)
    retrained_model = retrain_model(base_model, data)
    fine_tune_model(retrained_model, data)


def retrain_model(model: keras.Model, data: TrainDataset) -> keras.Model:
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    model.layers[6].trainable = True
    model.layers[-1].trainable = True
    model.compile(keras.optimizers.Adam(learning_rate=c.learning_rate))
    history: tf.keras.callbacks.History = model.fit(data.train_dataset,
                                                    epochs=150,
                                                    validation_data=data.validation_dataset)
    save_models(model, 'retrained')
    return model


def fine_tune_model(model: keras.Model, data: TrainDataset) -> None:
    model.trainable = True
    model.compile(keras.optimizers.Adam(learning_rate=1e-5))
    history_fine_tune: tf.keras.callbacks.History = model.fit(data.train_dataset,
                                                              epochs=100,
                                                              validation_data=data.validation_dataset)
    save_models(model, 'fine_tuned')


def save_models(model: keras.Model, name: str) -> None:
    model.save(Path(save_folder, f'{NAME}-{name}-full_model.h5'))
    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=c.learning_rate))
    prediction_model.save(Path(save_folder, f'{NAME}-{name}.h5'))


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Please specify 3 arguments: 1) a config file, 2) a path to a base model, ' +\
                               '& 3) a training run name.'
    c = TrainerConfiguration(Path(sys.argv[1]))
    base_model_path = Path(sys.argv[2])
    assert base_model_path.absolute().exists(), f'{base_model_path} is not a valid path.'
    NAME = sys.argv[3]

    save_folder = Path('./saved_models')
    if not save_folder.exists():
        os.makedirs(save_folder)

    try:
        gpu = gpu_selection()
        if gpu:
            with tf.device(f'/device:GPU:{gpu}'):
                print(f'Running on GPU {gpu}.')
                main()
        else:
            main()
    except Exception as e:
        print(e)
        tf.keras.backend.clear_session()
        exit(0)
