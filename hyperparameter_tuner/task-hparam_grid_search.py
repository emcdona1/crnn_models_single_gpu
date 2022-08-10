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
from utilities import Model, TrainerConfiguration
from utilities import TrainDataset
from utilities import gpu_selection


def train_test_tensorboard(hparams: dict, dataset: TrainDataset):
    model = Model(c)
    model.create_model(hparams[HP_KERNEL_SIZE], 'relu',
                         hparams[HP_NUM_DENSE_UNITS1], hparams[HP_DROPOUT],
                         hparams[HP_NUM_DENSE_LSTM1], 1024, hparams[HP_LEARNING_RATE])
    history = model.model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                        epochs=c.num_epochs)
    # _, accuracy = model.model.evaluate(dataset.train_dataset, dataset.validation_dataset)
    # return accuracy
    tuning_metric = history.history['val_loss'][-1]
    return tuning_metric


def run(run_dir, hparams: dict, dataset: TrainDataset):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        val_loss = train_test_tensorboard(hparams, dataset)
        tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=1)


def main():
    dataset = TrainDataset(c)
    dataset.create_dataset(4)
    log_folder_name = 'logs/hparam_tuning_grid'
    if Path(log_folder_name).exists():
        shutil.rmtree(log_folder_name)
    with tf.summary.create_file_writer(log_folder_name).as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_KERNEL_SIZE, HP_NUM_DENSE_UNITS1,
                     HP_DROPOUT, HP_NUM_DENSE_LSTM1, HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_VAL_LOSS, display_name='Validation Loss')]
        )
    session_num = 0
    for batch in HP_BATCH_SIZE.domain.values:
        for kernel in HP_KERNEL_SIZE.domain.values:
            for dense_1 in HP_NUM_DENSE_UNITS1.domain.values:
                for dropout in HP_DROPOUT.domain.values:
                    for ltsm_1 in HP_NUM_DENSE_LSTM1.domain.values:
                        for lr in HP_LEARNING_RATE.domain.values:
                            dataset.update_batch_size(batch)
                            hparams = {
                                HP_BATCH_SIZE: batch,
                                HP_KERNEL_SIZE: kernel,
                                HP_NUM_DENSE_UNITS1: dense_1,
                                HP_DROPOUT: dropout,
                                HP_NUM_DENSE_LSTM1: ltsm_1,
                                HP_LEARNING_RATE: lr
                            }
                            run_name = f'run-{session_num}'
                            print(f'--- Starting trial: {run_name}')
                            print({h.name: hparams[h] for h in hparams})
                            run(log_folder_name + run_name, hparams, dataset)
                            session_num += 1


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1]).absolute()
    assert config_location.is_file(), f'{config_location} is not a file.'
    c = TrainerConfiguration(config_location)
    tf.random.set_seed(c.seed)
    METRIC_VAL_LOSS = 'val_loss'

    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 128]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3]))  # 3, 4
    HP_NUM_DENSE_UNITS1 = hp.HParam('num_dense_units1', hp.Discrete([128, 256, 512]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.3, 0.4]))
    HP_NUM_DENSE_LSTM1 = hp.HParam('num_dense_lstm1', hp.Discrete([256, 512, 768, 1024]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0005, 0.005, 0.01, 0.05, 0.1]))

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
