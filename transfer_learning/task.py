import sys
import os
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from utilities import TrainDataset
from utilities import TrainerConfiguration


def main():
    base_model = keras.models.load_model(base_model_path)
    data = TrainDataset(c)
    data.create_dataset(c.batch_size, c.image_set_location, c.metadata_file_name)
    retrained_model = retrain_model(base_model, data)
    fine_tuned_model = fine_tune_model(retrained_model, data)


def retrain_model(model: keras.Model, data: TrainDataset):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    model.layers[6].trainable = True
    model.layers[-1].trainable = True
    model.compile(keras.optimizers.Adam(learning_rate=c.learning_rate))

    history = model.fit(data.train_dataset, epochs=100, validation_data=data.validation_dataset)

    model.save(Path(save_folder, 'retrained-full_model'))

    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=c.learning_rate))
    prediction_model.save(Path(save_folder, 'retrained'))
    return model


def fine_tune_model(model: keras.Model, data: TrainDataset):
    # Then, fine tune the model
    model.trainable = True
    model.compile(keras.optimizers.Adam(learning_rate=1e-5))
    history_fine_tune = model.fit(data.train_dataset, epochs=100, validation_data=data.validation_dataset)

    model.save(Path(save_folder, 'fine_tuned-full_model'))

    prediction_model = tf.keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense_layer').output
    )
    prediction_model.compile(tf.keras.optimizers.Adam(learning_rate=c.learning_rate))
    prediction_model.save(Path(save_folder, 'fine_tuned'))
    return model


if __name__ == '__main__':
    c = TrainerConfiguration(Path(sys.argv[1]))

    base_model_path = Path(sys.argv[2])
    assert base_model_path.absolute().exists(), f'{base_model_path} is not a valid path.'
    save_folder = Path('./saved_models')
    if not save_folder.exists():
        os.makedirs(save_folder)

    main()