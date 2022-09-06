import argparse
from pathlib import Path
import os
import sys
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)
from utilities import TrainerConfiguration


class ModelArguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser('Train one handwriting model.')
        self._args: argparse.Namespace = self.set_up_parser_arguments()
        self.run_name: str = self._args.run_name
        self.config: TrainerConfiguration = self._validate_config_location()
        self.lr: float = self._validate_learning_rate()
        self.batch_size: int = self._validate_batch_size()
        self.kernel_size: int = self._validate_kernel_size()
        self.activation: str = self._validate_activation_function()
        self.num_units_dense: int = self._validate_dense()
        self.dropout: float = self._validate_dropout()
        self.num_units_lstm1: int = self._validate_lstm1()
        self.num_units_lstm2: int = self._validate_lstm2()

    def set_up_parser_arguments(self):
        self._parser.add_argument('config', help='Config file location.')
        self._parser.add_argument('run_name', help='A name for this training run.')
        self._parser.add_argument('-lr', '--learning_rate', type=float, help='Desired learning rate (0, 1)')
        self._parser.add_argument('-b', '--batch_size', type=int, help='Desired batch size [1, ?)')
        self._parser.add_argument('-k', '--kernel_size', type=int,
                                  help='Kernel size for convolutions [2, ?)')
        self._parser.add_argument('-a', '--activation',
                                  help='Activation function for convolutions.')
        self._parser.add_argument('-dense', '--num_units_dense', type=int,
                                  help='# of units in fully connected layer, [2, ?)')
        self._parser.add_argument('-d', '--dropout', default=0.1, type=float,
                                  help='Dropout rate [0, 1), default = 0.1.')
        self._parser.add_argument('-lstm1', '--num_units_lstm1', type=int,
                                  help='# of units in first LSTM layer, [100, ?)')
        self._parser.add_argument('-lstm2', '--num_units_lstm2', type=int,
                                  help='# of units in first LSTM layer, [100, ?)')
        return self._parser.parse_args()

    def _validate_config_location(self) -> TrainerConfiguration:
        config_location = Path(self._args.config).absolute()
        assert config_location.is_file(), f'{config_location} is not a file.'
        return TrainerConfiguration(config_location)

    def _validate_learning_rate(self) -> float:
        if self._args.learning_rate:
            lr = self._args.learning_rate
        else:
            lr = self.config.learning_rate
        assert 0 < lr < 1, f'Learning rate {lr:.6f} is not valid. Must be in range (0, 1).'
        return lr

    def _validate_batch_size(self) -> int:
        if self._args.batch_size:
            batch_size = self._args.batch_size
        else:
            batch_size = self.config.batch_size
        assert batch_size >= 2, f'{batch_size} is not a valid batch size. Must be >= 2.'
        return batch_size

    def _validate_kernel_size(self) -> int:
        if self._args.kernel_size:
            kernel = self._args.kernel_size
        else:
            kernel = self.config.kernel_size
        assert kernel >= 2, f'{kernel} is not a valid kernel size.  Must be >= 2.'
        return kernel

    def _validate_activation_function(self) -> str:
        if self._args.activation:
            return self._args.activation
        else:
            return self.config.activation_function

    def _validate_dense(self) -> int:
        if self._args.num_units_dense:
            dense = self._args.num_units_dense
        else:
            dense = self.config.num_units_dense
        assert dense >= 2, f'{dense} is not a valid dense layer size.  Must be >= 2.'
        return dense

    def _validate_dropout(self) -> float:
        if self._args.dropout:
            dropout = self._args.dropout
        else:
            dropout = self.config.dropout
        assert 0 <= dropout < 1, f'{dropout} is not a valid dropout layer size.  Must be [0, 1).'
        return dropout

    def _validate_lstm1(self) -> int:
        if self._args.num_units_lstm1:
            num_units_lstm1 = self._args.num_units_lstm1
        else:
            num_units_lstm1 = self.config.num_units_lstm1
        assert num_units_lstm1 >= 2, f'{num_units_lstm1} is not a valid LSTM layer size.  Must be >= 100.'
        return num_units_lstm1

    def _validate_lstm2(self) -> int:
        if self._args.num_units_lstm2:
            num_units_lstm2 = self._args.num_units_lstm2
        else:
            num_units_lstm2 = self.config.num_units_lstm2
        assert num_units_lstm2 >= 2, f'{num_units_lstm2} is not a valid kernel size.  Must be >= 100.'
        return num_units_lstm2
