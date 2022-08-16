import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress oneDNN INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

from pathlib import Path
import tensorflow as tf
import pandas as pd
from utilities import Model
from utilities import TrainDataset
from utilities import gpu_selection
from arguments import ModelArguments


def main():
    dataset = TrainDataset(arg.config_location)
    dataset.create_dataset(arg.batch_size if arg.batch_size else dataset.c.batch_size)
    tf.random.set_seed(dataset.c.seed)
    model = Model(dataset.c)
    model.create_model(arg.kernel_size,
                       arg.activation,
                       arg.num_units_dense,
                       arg.dropout,
                       arg.num_units_lstm1,
                       arg.num_units_lstm2,
                       arg.lr if arg.lr else dataset.c.learning_rate)
    history = model.model.fit(dataset.train_dataset, epochs=dataset.c.num_epochs,
                              validation_data=dataset.validation_dataset)

    results_folder = Path('saved_models')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    training_history = pd.DataFrame(history.history)
    history_filename = f'{NAME}-training_history.csv'
    training_history.to_csv(Path(results_folder, history_filename))

    training_model_name = f'{NAME}-full_model'
    save_location = Path(results_folder, f'{training_model_name}.h5')
    model.model.save(save_location)
    print(f'Training model saved to: {save_location}')

    # create and save prediction model
    prediction_model = tf.keras.models.Model(
        model.model.get_layer(name='image').input, model.model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=arg.lr if arg.lr else dataset.c.learning_rate))
    prediction_model_name = f'{NAME}-prediction'
    save_location = Path(results_folder, f'{prediction_model_name}.h5')
    prediction_model.save(save_location)
    print(f'Prediction model saved to: {save_location}')


if __name__ == '__main__':
    arg = ModelArguments()
    NAME = arg.run_name
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
