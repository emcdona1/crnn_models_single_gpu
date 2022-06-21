from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
      

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

CHAR_LIST: str = '\' !"#&()[]*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
characters = sorted(set(list(CHAR_LIST)))
# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
vocabulary=list(characters), mask_token=None
)

def create_model(kernel_size: int, activation: str, num_units_dense1: int, dropout: float, num_units_ltsm1: int, num_units_ltsm2: int, learning_rate: float):
    '''Defines and compiles model.'''
    # Inputs to the model
    input_img = layers.Input(shape=(400, 100, 1), name='image', dtype='float32')  # TODO: don't hardcode this
    labels = layers.Input(name='label', shape=(None,), dtype='float32')

    # First conv block
    x = layers.Conv2D(32, (kernel_size, kernel_size), activation=activation, 
                      kernel_initializer='he_normal', padding='same', 
                      name='Conv1')(input_img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Second conv block
    x = layers.Conv2D(64, (kernel_size, kernel_size), activation=activation, 
                      kernel_initializer='he_normal', padding='same', 
                      name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((400 // 4), (100 // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(num_units_dense1, activation=activation, name='dense1')(x)
    x = layers.Dropout(dropout)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(num_units_ltsm1, return_sequences=True, dropout=0.25, name='bidirection_ltsm_1'))(x)
    x = layers.Bidirectional(layers.LSTM(num_units_ltsm2, return_sequences=True, dropout=0.25, name='bidirection_ltsm_1'))(x)

    # Output layer
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation='softmax', name='dense_layer')(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name='ocr_model_v1')
    # Compile the model and return
    model.compile(keras.optimizers.Adam(learning_rate=learning_rate))
    print(model.summary())
    return model
