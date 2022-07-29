import sys
import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress INFO alerts about TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # suppress INFO alerts about oneDNN
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

import tensorflow as tf
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from tensorboard.plugins.hparams import api as hp
from utilities import create_model
from utilities import TrainDataset


def train_test_tensorboard(hparams: dict, dataset: TrainDataset):
    model = create_model(hparams[HP_KERNEL_SIZE], 'relu',
                         hparams[HP_NUM_DENSE_UNITS1], hparams[HP_DROPOUT],
                         hparams[HP_NUM_DENSE_LTSM1], 1024,
                         hparams[HP_LEARNING_RATE])
    history = model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                        epochs=10,  # todo: change to epochs=c.NUM_EPOCHS after testing
                        callbacks=[TuneReportCallback({'mean_loss': 'val_loss'})])
    # _, accuracy = model.evaluate(dataset.train_dataset, dataset.validation_dataset)
    # return accuracy
    tuning_metric = history.history['val_loss'][-1]
    return tuning_metric


def run(run_dir, hparams: dict, dataset: TrainDataset):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        val_loss = train_test_tensorboard(hparams, dataset)
        tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=1)


def tensorboard_grid_search(dataset):
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_KERNEL_SIZE, HP_NUM_DENSE_UNITS1,
                     HP_DROPOUT, HP_NUM_DENSE_LTSM1, HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_LOSS, display_name='Loss'),
                     hp.Metric(METRIC_VAL_LOSS, display_name='Validation Loss')],
        )
    session_num = 0
    for batch in HP_BATCH_SIZE.domain.values:
        for kernel in HP_KERNEL_SIZE.domain.values:
            for dense_1 in HP_NUM_DENSE_UNITS1.domain.values:
                for dropout in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    for ltsm_1 in (HP_NUM_DENSE_LTSM1.domain.min_value, HP_NUM_DENSE_LTSM1.domain.max_value):
                        for lr in (HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value):
                            dataset.update_batch_size(batch)
                            hparams = {
                                HP_BATCH_SIZE: batch,
                                HP_KERNEL_SIZE: kernel,
                                HP_NUM_DENSE_UNITS1: dense_1,
                                HP_DROPOUT: dropout,
                                HP_NUM_DENSE_LTSM1: ltsm_1,
                                HP_LEARNING_RATE: lr
                            }
                            run_name = f'run-{session_num}'
                            print(f'--- Starting trial: {run_name}')
                            print({h.name: hparams[h] for h in hparams})
                            run('logs/hparam_tuning/' + run_name, hparams, dataset)
                            session_num += 1


def train_ray(config, checkpoint_dir=None):
    dataset = TrainDataset(config_location)
    dataset.create_dataset(config['batch_size'])
    model = create_model(kernel_size=config['kernel_size'],
                         activation='relu',
                         num_units_dense1=config['num_dense_units1'],
                         dropout=config['dropout'],
                         num_units_ltsm1=config['num_dense_ltsm1'],
                         num_units_ltsm2=1024,
                         learning_rate=config['learning_rate'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "model.h5", monitor='loss', save_best_only=True, save_freq=2)
    history = model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                        epochs=15,
                        verbose=0,
                        callbacks=[checkpoint_callback, TuneReportCallback({'validation_loss': 'val_loss'})])
    print(history)


def ray_hyperband_search():
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
            'num_dense_ltsm1': tune.randint(256, 2048),
            'learning_rate': tune.uniform(0.0005, 0.1)
        },
    )
    print(f'Best hyperparameters found were: {analysis.best_config}')


def main():
    ray_hyperband_search()
    # tensorboard_grid_search()


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'

    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256]))  # 32
    # HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 4, 5])) # 3, 4
    # HP_NUM_DENSE_UNITS1 = hp.HParam('num_dense_units1', hp.Discrete([128, 256, 512]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(min_value=0.1, max_value=0.5))
    # HP_NUM_DENSE_LTSM1 = hp.HParam('num_dense_ltsm1', hp.IntInterval(min_value=256, max_value=2048))
    # HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(min_value=0.0005, max_value=0.1))
    # METRIC_LOSS = 'loss'
    # METRIC_VAL_LOSS = 'val_loss'

    main()

    # View results in iPython:
    # %load_ext tensorboard
    # %tensorboard --logdir logs/hparam_tuning
    # View results from console:
    # tensorboard --logdir logs/hparam_tuning
