import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress INFO alert about oneDNN
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from utilities import create_model
from utilities import TrainDataset


def train_test_model(hparams: dict, dataset: TrainDataset):
    model = create_model(hparams[HP_KERNEL_SIZE], hparams[HP_ACTIVATION],
                         hparams[HP_NUM_DENSE_UNITS1], hparams[HP_DROPOUT],
                         hparams[HP_NUM_DENSE_LTSM1], hparams[HP_NUM_DENSE_LTSM2],
                         hparams[HP_LEARNING_RATE])
    history = model.fit(dataset.train_dataset, validation_data=dataset.validation_dataset,
                        epochs=10)  # todo: change to epochs=c.NUM_EPOCHS after testing
    # _, accuracy = model.evaluate(dataset.train_dataset, dataset.validation_dataset)
    # return accuracy
    tuning_metric = history.history['val_loss'][-1]
    return tuning_metric


def run(run_dir, hparams: dict, dataset: TrainDataset):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        val_loss = train_test_model(hparams, dataset)
        tf.summary.scalar(METRIC_VAL_LOSS, val_loss, step=1)


def main():
    assert len(sys.argv) >= 2, 'Please specify a config file.'
    config_location = Path(sys.argv[1])
    dataset = TrainDataset(config_location)
    dataset.create_dataset(32)

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_KERNEL_SIZE, HP_ACTIVATION, HP_NUM_DENSE_UNITS1,
                     HP_DROPOUT, HP_NUM_DENSE_LTSM1, HP_NUM_DENSE_LTSM2, HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_LOSS, display_name='Loss'),
                     hp.Metric(METRIC_VAL_LOSS, display_name='Validation Loss')],
        )

    session_num = 0
    for batch in HP_BATCH_SIZE.domain.values:
        for kernel in HP_KERNEL_SIZE.domain.values:
            for activation in HP_ACTIVATION.domain.values:
                for dense_1 in HP_NUM_DENSE_UNITS1.domain.values:
                    for dropout in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                        for ltsm_1 in HP_NUM_DENSE_LTSM1.domain.values:
                            for ltsm_2 in HP_NUM_DENSE_LTSM2.domain.values:
                                for lr in (HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value):
                                    dataset.update_batch_size(batch)
                                    hparams = {
                                        HP_BATCH_SIZE: batch,
                                        HP_KERNEL_SIZE: kernel,
                                        HP_ACTIVATION: activation,
                                        HP_NUM_DENSE_UNITS1: dense_1,
                                        HP_DROPOUT: dropout,
                                        HP_NUM_DENSE_LTSM1: ltsm_1,
                                        HP_NUM_DENSE_LTSM2: ltsm_2,
                                        HP_LEARNING_RATE: lr
                                    }
                                    run_name = f'run-{session_num}'
                                    print(f'--- Starting trial: {run_name}')
                                    print({h.name: hparams[h] for h in hparams})
                                    run('logs/hparam_tuning/' + run_name, hparams, dataset)
                                    session_num += 1


if __name__ == "__main__":
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64, 128]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([2, 3, 4]))
    HP_ACTIVATION = hp.HParam('activation_function', hp.Discrete(['relu', 'sigmoid', 'tanh']))
    HP_NUM_DENSE_UNITS1 = hp.HParam('num_dense_units1', hp.Discrete([64, 128, 256]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(min_value=0.1, max_value=0.5))
    HP_NUM_DENSE_LTSM1 = hp.HParam('num_dense_ltsm1', hp.Discrete([128, 256, 512, 768, 1024]))
    HP_NUM_DENSE_LTSM2 = hp.HParam('num_dense_ltsm2', hp.Discrete([128, 256, 512, 768, 1024]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(min_value=0.0005, max_value=0.1))
    METRIC_LOSS = 'loss'
    METRIC_VAL_LOSS = 'val_loss'

    main()
