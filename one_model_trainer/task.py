import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress oneDNN INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

from pathlib import Path
import tensorflow as tf
import pandas as pd
from utilities import create_model, CTCLayer
from utilities import TrainDataset
from utilities import gpu_selection


def main():
    ###################################################################
    # SET ALL THE HYPERPARAMETERS HERE, WHICH WERE DETERMINED IN TUNING
    kernel_size = 4
    activation_function = 'relu'
    learning_rate = 0.001
    num_units_dense = 256
    num_units_ltsm2 = 1024
    ###################################################################
    # run 55 parameters
    dropout = 0.12489316869910207
    num_units_ltsm1 = 512
    # run 41 parameters
    # dropout = 0.1
    # num_units_ltsm1 = 768
    ###################################################################

    dataset = TrainDataset(config_location)
    dataset.create_dataset(dataset.c.batch_size)
    tf.random.set_seed(dataset.c.seed)
    model = create_model(kernel_size, activation_function, num_units_dense, dropout,
                         num_units_ltsm1, num_units_ltsm2, learning_rate)
    history = model.fit(dataset.train_dataset, epochs=dataset.c.num_epochs, validation_data=dataset.validation_dataset)

    results_folder = Path('saved_models')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    training_history = pd.DataFrame(history.history)
    history_filename = f'{NAME}-training_history.csv'
    training_history.to_csv(Path(results_folder, history_filename))

    training_model_name = f'{NAME}-full_model'
    save_location = Path(results_folder, f'{training_model_name}.h5')
    model.save(save_location, custom_objects={'CTCLayer': CTCLayer})
    print(f'Training model saved to: {save_location}')

    # create and save prediction model
    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate))
    prediction_model_name = f'{NAME}-prediction'
    save_location = Path(results_folder, f'{prediction_model_name}.h5')
    prediction_model.save(save_location, custom_objects={'CTCLayer': CTCLayer})
    print(f'Prediction model saved to: {save_location}')

    
if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Please specify a config file & a training run name.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'
    NAME = sys.argv[2]

    gpu = gpu_selection()
    if gpu:
        with tf.device(f'/device:GPU:{gpu}'):
            print(f'Running on GPU {gpu}.')
            main()
    else:
        main()
