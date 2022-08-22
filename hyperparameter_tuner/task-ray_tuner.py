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
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from utilities import Model, TunerConfiguration
from utilities import TrainDataset
from utilities import gpu_selection


def train_ray(config, checkpoint_dir=None):
    dataset = TrainDataset(c)
    dataset.create_dataset(config['batch_size'])
    model = Model(c)
    model.create_model(kernel_size=config['kernel_size'],
                       activation='relu',
                       num_units_dense1=config['num_dense_units1'],
                       dropout=config['dropout'],
                       num_units_lstm1=config['num_dense_lstm1'],
                       num_units_lstm2=1024,
                       learning_rate=config['learning_rate'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "model.h5", monitor='loss', save_best_only=True, save_freq=2)
    history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                              epochs=15,
                              verbose=0,
                              callbacks=[checkpoint_callback, TuneReportCallback({'validation_loss': 'val_loss'})])
    print(history)


def main():
    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration', max_t=400, grace_period=20)
    analysis = tune.run(
        train_ray,
        name='iam_train',
        scheduler=scheduler,
        metric='validation_loss',
        mode='min',
        stop={'training_iteration': 10},
        num_samples=5,
        resources_per_trial={'cpu': 1, 'gpu': 0.2},  # to use one GPU total, 'gpu' = 1 / num_samples
        config={
            'threads': 2,
            'batch_size': tune.choice([32, 64, 158, 256]),
            'kernel_size': tune.randint(3, 5),
            'num_dense_units1': tune.randint(128, 512),
            'dropout': tune.uniform(0.1, 0.5),
            'num_dense_lstm1': tune.randint(256, 2048),
            'learning_rate': tune.uniform(0.0005, 0.1)
        },
    )
    print(f'Best hyperparameters found were: {analysis.best_config}')


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'
    c = TunerConfiguration(config_location)
    tf.random.set_seed(c.seed)

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
