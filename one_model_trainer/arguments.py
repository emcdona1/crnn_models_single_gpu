import argparse
from typing import Union
from pathlib import Path


class ModelArguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser('Train one handwriting model.')
        self._args: argparse.Namespace = self.set_up_parser_arguments()
        self.run_name = self._args.run_name
        self.config_location = self._validate_config_location()
        self.lr: float = self._validate_learning_rate()
        self.batch_size: int = self._validate_batch_size()
        self.kernel_size: int = self._validate_kernel_size()
        self.activation: str = self._validate_activation_function()
        self.num_units_dense: int = self._validate_dense()
        self.dropout: float = self._validate_dropout()
        self.num_units_lstm1: int = self._validate_lstm1()
        self.num_units_lstm2: int = self._validate_lstm2()

    def set_up_parser_arguments(self):
        # learning_rate = 0.0005
        # batch_size = 105
        # kernel_size = 5
        # activation_function = 'relu'
        # num_units_dense = 128
        # dropout = 0.12489316869910207
        # num_units_lstm1 = 512
        # num_units_lstm2 = 1024
        self._parser.add_argument('config', help='Config file location.')
        self._parser.add_argument('run_name', help='A name for this training run.')
        self._parser.add_argument('-lr', '--learning_rate', type=float, help='Desired learning rate (0, 1).')
        self._parser.add_argument('-b', '--batch_size', type=int, help='Desired batch size [1, ?).')
        self._parser.add_argument('-k', '--kernel_size', type=int, default=4,
                                  help='Kernel size for convolutions [2,?), default = 4.')
        self._parser.add_argument('-a', '--activation', default='relu',
                                  help='Activation function for convolutions, default = relu.')
        self._parser.add_argument('-dense', '--num_units_dense', type=int, default=256,
                                  help='# of units in fully connected layer, [2, ?), default = 256)')
        self._parser.add_argument('-d', '--dropout', default=0.1, type=float,
                                  help='Dropout rate [0, 1), default = 0.1.')
        self._parser.add_argument('-lstm1', '--num_units_lstm1', type=int, default=512,
                                  help='# of units in first LSTM layer, [100, ?), default = 512)')
        self._parser.add_argument('-lstm2', '--num_units_lstm2', type=int, default=512,
                                  help='# of units in first LSTM layer, [100, ?), default = 512)')
        return self._parser.parse_args()

    def _validate_config_location(self) -> Path:
        config_location = Path(self._args.config).absolute()
        assert config_location.is_file(), f'{config_location} is not a file.'
        return config_location

    def _validate_learning_rate(self) -> Union[float, None]:
        if self._args.learning_rate:
            lr = self._args.learning_rate
            assert 0 < lr < 1, f'Learning rate {lr:.6f} is not valid. Must be in range (0, 1).'
            return lr
        else:
            return None

    def _validate_batch_size(self) -> Union[int, None]:
        if self._args.batch_size:
            batch_size = self._args.batch_size
            assert batch_size >= 2, f'{batch_size} is not a valid batch size. Must be >= 2.'
            return batch_size
        else:
            return None

    def _validate_kernel_size(self) -> int:
        kernel = self._args.kernel_size
        assert kernel >= 2, f'{kernel} is not a valid kernel size.  Must be >= 2.'
        return kernel

    def _validate_activation_function(self) -> str:
        return self._args.activation

    def _validate_dense(self) -> int:
        dense = self._args.kernel_size
        assert dense >= 2, f'{dense} is not a valid dense layer size.  Must be >= 2.'
        return dense

    def _validate_dropout(self) -> float:
        dropout = self._args.dropout
        assert 0 <= dropout < 1, f'{dropout} is not a valid dropout layer size.  Must be [0, 1).'
        return dropout

    def _validate_lstm1(self) -> int:
        num_units_lstm1 = self._args.num_units_lstm1
        assert num_units_lstm1 >= 2, f'{num_units_lstm1} is not a valid LSTM layer size.  Must be >= 100.'
        return num_units_lstm1

    def _validate_lstm2(self) -> int:
        num_units_lstm2 = self._args.num_units_lstm2
        assert num_units_lstm2 >= 2, f'{num_units_lstm2} is not a valid kernel size.  Must be >= 100.'
        return num_units_lstm2
