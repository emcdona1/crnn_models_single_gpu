import os
from pathlib import Path
import argparse
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from trainer.create_dataset import create_dataset
from trainer.create_model import create_model
from trainer.trainer_config import TrainerConfiguration
c = TrainerConfiguration()
tf.random.set_seed(c.SEED)


###################################################################
# SET ALL THE HYPERPARAMETERS HERE, WHICH WERE DETERMINED IN TUNING
NAME = 'run_55_train'  # change the name
# NAME = 'run_41'
BATCH_SIZE=128
KERNEL_SIZE=4
ACTIVATION_FUNCTION='relu'
LEARNING_RATE=0.001
DROPOUT=0.12489316869910207
# DROPOUT=0.1
NUM_UNITS_DENSE=256
NUM_UNITS_LTSM1=512
# NUM_UNITS_LTSM1=768
NUM_UNITS_LTSM2=1024
###################################################################


def get_args(manual_args=None):
    '''Parses args. Must include all hyperparameters you want to specify.'''
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
    args = parser.parse_args() if not manual_args else parser.parse_args(manual_args)
    return args


def main():
    manual_args = [f'--batch_size={BATCH_SIZE}', f'--kernel_size={KERNEL_SIZE}', f'--activation={ACTIVATION_FUNCTION}', f'--dropout={DROPOUT}',
            f'--num_units_dense1={NUM_UNITS_DENSE}', f'--num_units_ltsm1={NUM_UNITS_LTSM1}', f'--num_units_ltsm2={NUM_UNITS_LTSM2}', f'--learning_rate={LEARNING_RATE}']
    args = get_args(manual_args)
    train_data, validation_data = create_dataset(args.batch_size)
    model = create_model(args.kernel_size, args.activation, args.num_units_dense1, args.dropout, 
                     args.num_units_ltsm1, args.num_units_ltsm2, args.learning_rate)
    history = model.fit(train_data, epochs=c.NUM_EPOCHS, validation_data=validation_data)



    results_folder = Path('results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    training_history = pd.DataFrame(model.history.history)
    history_filename = f'{NAME}-training_history.csv'
    training_history.to_csv(Path(results_folder, history_filename))

    training_model_name = f'{NAME}-trained'
    model.save(Path(results_folder, f'{training_model_name}.h5'))

    # create and save prediction model
    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    prediction_model_name = f'{NAME}-prediction'
    prediction_model.save(Path(results_folder, f'{prediction_model_name}.h5'))

    
if __name__ == '__main__':
    main()
