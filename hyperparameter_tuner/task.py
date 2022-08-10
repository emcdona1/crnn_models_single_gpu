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
# from ray import tune
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.integration.keras import TuneReportCallback
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
global dimensions
dimensions = [Integer(low=32, high=256, name='batch_size'), Integer(low=3, high=5, name='kernel_size'),
              Integer(low=128, high=512, name='num_dense_units1'), Real(low=0.1, high=0.5, name='dropout'),
              Integer(low=256, high=2048, name='num_dense_lstm1'),
              Integer(low=256, high=2048, name='num_dense_lstm2'),
              Real(low=0.0005, high=0.1, prior='log-uniform', name='learning_rate')]


def bayesian_search():
    global dataset
    dataset = TrainDataset(c)
    dataset.create_dataset(4, c.image_set_location, c.metadata_file_name)
    global dimensions
    search_result: OptimizeResult = gp_minimize(func=fit_new_model,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement
                                n_calls=3,
                                random_state=c.seed,
                                n_random_starts=2)
    minima: list = search_result.x
    print(f'Lowest validation loss: {search_result.fun:.4f}')
    print(f'''Best values found:
    - batch_size: {minima[0]}
    - kernel_size: {minima[1]},
    = num_dense_units1: {minima[2]},
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
    model = Model(c)
    model.create_model(params['kernel_size'], 'relu', params['num_dense_units1'], params['dropout'], params['num_dense_lstm1'],
                  params['num_dense_lstm2'], params['learning_rate'])
    history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                            epochs=c.num_epochs,
                            # callbacks=[TuneReportCallback({'mean_loss': 'val_loss'})]
    )
    tuning_metric = history.history['val_loss'][-1]
    return tuning_metric


# def train_test_tensorboard(hparams: dict, dataset: TrainDataset):
#     model = Model(c)
#     model.create_model(hparams[HP_KERNEL_SIZE], 'relu',
#                          hparams[HP_NUM_DENSE_UNITS1], hparams[HP_DROPOUT],
#                          hparams[HP_NUM_DENSE_LSTM1], 1024, hparams[HP_LEARNING_RATE])
#     history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
#                         epochs=c.num_epochs,
#                         callbacks=[TuneReportCallback({'mean_loss': 'val_loss'})])
#     # _, accuracy = model.model.evaluate(dataset.train_dataset, dataset.validation_dataset)
#     # return accuracy
#     tuning_metric = history.history['val_loss'][-1]
#     return tuning_metric
#
#
# def run(run_dir, hparams: dict, dataset: TrainDataset):
#     with tf.summary.create_file_writer(run_dir).as_default():
#         hp.hparams(hparams)  # record the values used in this trial
#         val_loss = train_test_tensorboard(hparams, dataset)
#         tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=1)
#
#
# def tensorboard_grid_search():
#     dataset = TrainDataset(c)
#     dataset.create_dataset(4)
#     log_folder_name = 'logs/hparam_tuning_grid'
#     if Path(log_folder_name).exists():
#         shutil.rmtree(log_folder_name)
#     with tf.summary.create_file_writer(log_folder_name).as_default():
#         hp.hparams_config(
#             hparams=[HP_BATCH_SIZE, HP_KERNEL_SIZE, HP_NUM_DENSE_UNITS1,
#                      HP_DROPOUT, HP_NUM_DENSE_LSTM1, HP_LEARNING_RATE],
#             metrics=[hp.Metric(METRIC_VAL_LOSS, display_name='Validation Loss')]
#         )
#     session_num = 0
#     for batch in HP_BATCH_SIZE.domain.values:
#         for kernel in HP_KERNEL_SIZE.domain.values:
#             for dense_1 in HP_NUM_DENSE_UNITS1.domain.values:
#                 for dropout in HP_DROPOUT.domain.values:
#                     for ltsm_1 in HP_NUM_DENSE_LSTM1.domain.values:
#                         for lr in HP_LEARNING_RATE.domain.values:
#                             dataset.update_batch_size(batch)
#                             hparams = {
#                                 HP_BATCH_SIZE: batch,
#                                 HP_KERNEL_SIZE: kernel,
#                                 HP_NUM_DENSE_UNITS1: dense_1,
#                                 HP_DROPOUT: dropout,
#                                 HP_NUM_DENSE_LSTM1: ltsm_1,
#                                 HP_LEARNING_RATE: lr
#                             }
#                             run_name = f'run-{session_num}'
#                             print(f'--- Starting trial: {run_name}')
#                             print({h.name: hparams[h] for h in hparams})
#                             run(log_folder_name + run_name, hparams, dataset)
#                             session_num += 1
#
#
# def train_ray(config, checkpoint_dir=None):
#     dataset = TrainDataset(config_location)
#     dataset.create_dataset(config['batch_size'])
#     model = Model(c)
#     model.create_model(kernel_size=config['kernel_size'],
#                          activation='relu',
#                          num_units_dense1=config['num_dense_units1'],
#                          dropout=config['dropout'],
#                          num_units_lstm1=config['num_dense_lstm1'],
#                          num_units_lstm2=1024,
#                          learning_rate=config['learning_rate'])
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         "model.h5", monitor='loss', save_best_only=True, save_freq=2)
#     history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
#                         epochs=15,
#                         verbose=0,
#                         callbacks=[checkpoint_callback, TuneReportCallback({'validation_loss': 'val_loss'})])
#     print(history)
#
#
# def ray_hyperband_search():
#     scheduler = AsyncHyperBandScheduler(time_attr='training_iteration', max_t=400, grace_period=20)
#     analysis = tune.run(
#         train_ray,
#         name='iam_train',
#         scheduler=scheduler,
#         metric='validation_loss',
#         mode='min',
#         stop={'training_iteration': 10},
#         num_samples=5,
#         resources_per_trial={'cpu': 1, 'gpu': 0.2},  # to use one GPU total, 'gpu' = 1 / num_samples
#         config={
#             'threads': 2,
#             'batch_size': tune.choice([32, 64, 158, 256]),
#             'kernel_size': tune.randint(3, 5),
#             'num_dense_units1': tune.randint(128, 512),
#             'dropout': tune.uniform(0.1, 0.5),
#             'num_dense_lstm1': tune.randint(256, 2048),
#             'learning_rate': tune.uniform(0.0005, 0.1)
#         },
#     )
#     print(f'Best hyperparameters found were: {analysis.best_config}')


def main():
    # ray_hyperband_search()
    # tensorboard_grid_search()
    bayesian_search()


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'
    c = TrainerConfiguration(config_location)
    tf.random.set_seed(c.seed)
    METRIC_VAL_LOSS = 'val_loss'

    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 128]))
    # HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3]))  # 3, 4
    # HP_NUM_DENSE_UNITS1 = hp.HParam('num_dense_units1', hp.Discrete([128, 256, 512]))
    # HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.3, 0.4]))
    # HP_NUM_DENSE_LSTM1 = hp.HParam('num_dense_lstm1', hp.Discrete([256, 512, 768, 1024]))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0005, 0.005, 0.01, 0.05, 0.1]))

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
