import sys
import os
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

from pathlib import Path
import tensorflow as tf
import pandas as pd
from utilities import TestConfiguration
from utilities import TestDataset
from utilities import CTCLayer
config = TestConfiguration(Path('../setup_tester.cfg'))


def main():
    test_dataset = TestDataset(config)
    test_dataset.create_dataset(32)
    prediction_model = tf.keras.models.load_model(config.model_file, custom_objects={'CTCLayer': CTCLayer})
    prediction_model.compile(optimizer=tf.keras.optimizers.Adam())

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
    print(prediction_results)

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    prediction_results.to_csv(Path('predictions', f'{config.model_file.stem}-predictions.csv'))


if __name__ == '__main__':
    main()
