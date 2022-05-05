from pathlib import Path
from tensorflow.keras import layers
import tensorflow as tf
import argparse
import hypertune
from trainer.create_model import create_model
from trainer.create_dataset import create_dataset
from trainer.trainer_config import TrainerConfiguration


def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--batch_size',
      required=True,
      type=int,
      help='batch size')
    parser.add_argument(
      '--kernel_size',
      required=True,
      type=int,
      help='size of kernel to use in convolutional layers (nxn square)')
    parser.add_argument(
      '--activation',
      required=True,
      type=str,
      help='activation function to use')
    parser.add_argument(
      '--dropout',
      required=True,
      type=float,
      help='amount of dropout in dropout layer')
    parser.add_argument(
      '--num_units_dense1',
      required=True,
      type=int,
      help='number of units in 1st dense layer')
    parser.add_argument(
      '--num_units_ltsm1',
      required=True,
      type=int,
      help='number of units in 1st LTSM bidirectional layer')
    parser.add_argument(
      '--num_units_ltsm2',
      required=True,
      type=int,
      help='number of units in 2nd LTSM bidirectional layer')
    parser.add_argument(
      '--learning_rate',
      required=True,
      type=float,
      help='learning rate')
    args = parser.parse_args()
    return args


def main():
    c = TrainerConfiguration()
    tf.random.set_seed(c.SEED)
    args = get_args()
    train_data, validation_data = create_dataset(args.batch_size)
    model = create_model(args.kernel_size, args.activation, args.num_units_dense1, args.dropout, 
                         args.num_units_ltsm1, args.num_units_ltsm2, args.learning_rate)
    history = model.fit(train_data, epochs=c.NUM_EPOCHS, validation_data=validation_data)

    # DEFINE METRIC
    tuning_metric = history.history['val_loss'][-1]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='val_loss',
        metric_value=tuning_metric,
        global_step=c.NUM_EPOCHS)


if __name__ == "__main__":
    main()
