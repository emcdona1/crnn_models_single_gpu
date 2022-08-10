import sys
import os
import shutil
import traceback
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress INFO alerts about TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # suppress INFO alerts about oneDNN
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from utilities import Model, TrainerConfiguration
from utilities import TrainDataset
from utilities import gpu_selection
from scipy.optimize import OptimizeResult
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt


global dataset
dimensions = [Integer(low=32, high=256, name='batch_size'), Integer(low=3, high=5, name='kernel_size'),
              Integer(low=128, high=512, name='num_dense_units1'), Real(low=0.1, high=0.5, name='dropout'),
              Integer(low=256, high=2048, name='num_dense_lstm1'),
              Integer(low=256, high=2048, name='num_dense_lstm2'),
              Real(low=0.0005, high=0.1, prior='log-uniform', name='learning_rate')]
session_num = 0
log_folder_name = Path('logs/hparam_tuning_bayes')


def main():
    global dataset
    dataset = TrainDataset(c)
    dataset.create_dataset(4, c.image_set_location, c.metadata_file_name)
    if Path(log_folder_name).exists():
        shutil.rmtree(log_folder_name)
    global dimensions
    search_result: OptimizeResult = gp_minimize(func=fit_new_model,
                                                dimensions=dimensions,
                                                acq_func='EI',  # Expected Improvement
                                                n_calls=20,
                                                random_state=c.seed,
                                                n_random_starts=3
                                                )
    minima: list = search_result.x
    print(f'Lowest validation loss: {search_result.fun:.4f}')
    print(f'''Best values found:
    - batch_size: {minima[0]}
    - kernel_size: {minima[1]},
    - num_dense_units1: {minima[2]},
    - dropout: {minima[3]},
    - num_dense_lstm1: {minima[4]},
    - num_dense_lstm2: {minima[5]},
    - learning_rate: {minima[6]}''')
    plot_convergence(search_result)
    plt.plot()
    plt.show()


@use_named_args(dimensions=dimensions)
def fit_new_model(**params):
    global dataset
    dataset.update_batch_size(params['batch_size'])
    global session_num
    model = Model(c)
    model.create_model(params['kernel_size'], 'relu', params['num_dense_units1'], params['dropout'],
                       params['num_dense_lstm1'],
                       params['num_dense_lstm2'], params['learning_rate'])
    global log_folder_name
    log_name = f"{params['batch_size']}-{params['kernel_size']}-{params['num_dense_units1']}-{params['dropout']:0.4f}-" + \
               f"{params['num_dense_lstm1']}-{params['num_dense_lstm2']}-{params['learning_rate']:.0e}"
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=Path(log_folder_name, log_name),
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False
    )
    history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                              epochs=c.num_epochs,
                              callbacks=[tb_callback]
                              )
    tuning_metric = history.history['val_loss'][-1]
    run_name = f'run-{session_num}'
    session_num += 1
    return tuning_metric


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'
    c = TrainerConfiguration(config_location)
    tf.random.set_seed(c.seed)
    METRIC_VAL_LOSS = 'val_loss'

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
        print(traceback.format_exc())
        tf.keras.backend.clear_session()
        exit(0)
