import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # suppress oneDNN INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
working_dir = os.path.join(os.getcwd())
sys.path.append(working_dir)

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from utilities import TestConfiguration
from utilities import TestDataset
from utilities import CTCLayer, gpu_selection


def main():
    c = TestConfiguration(config_location)
    test_dataset = TestDataset(c)
    test_dataset.create_dataset(32)

    for model_file in model_list:
        prediction_model = tf.keras.models.load_model(model_file, custom_objects={'CTCLayer': CTCLayer})
        prediction_model.compile(optimizer=tf.keras.optimizers.Adam())
        prediction_results = generate_predictions(prediction_model, test_dataset)
        save_predictions(model_file, prediction_results)


def generate_predictions(prediction_model: keras.Model, test_dataset: TestDataset) -> pd.DataFrame:
    prediction_results = pd.DataFrame(columns=['label', 'prediction'])
    for batch in test_dataset.test_dataset:
        images = batch['image']
        labels = batch['label']
        preds = prediction_model.predict(images)
        pred_texts = test_dataset.decode_batch_predictions(preds)
        pred_texts = [t.replace('[UNK]', '').replace(' ', '') for t in pred_texts]
        orig_texts = []
        for label in labels:
            label = tf.strings.reduce_join(test_dataset.num_to_char(label)).numpy().decode('utf-8')
            orig_texts.append(label)
        orig_texts = [t.replace('[UNK]', '').strip() for t in orig_texts]
        new_results = pd.DataFrame(zip(orig_texts, pred_texts), columns=['label', 'prediction'])
        prediction_results = prediction_results.append(new_results, ignore_index=True)
    return prediction_results


def save_predictions(model_file: Path, prediction_results: pd.DataFrame) -> None:
    save_folder = Path('./test_set_predictions/predictions')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = Path(save_folder, f'{model_file.stem}-predictions.csv')
    prediction_results.to_csv(save_path)
    print(f'Saved {save_path}')


if __name__ == '__main__':
    assert len(sys.argv) >= 3, 'Please specify a config file, and 1 or more model files.'
    config_location = Path(sys.argv[1]).absolute()
    model_list = [Path(m) for m in sys.argv[2:]]

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
        tf.keras.backend.clear_session()
        exit(0)
