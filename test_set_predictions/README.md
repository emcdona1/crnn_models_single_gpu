# Testing a model on a dataset

1. Run through the `generate_predictions.ipynb` Jupyter notebook within your Vertex AI workbench.  (If you are testing multiple models on the same test image set, you only need to download the test image set once.)
Results will be saved as CSV files in the `predictions` folder, including the ground truth text and the generated prediction text.

(Note: If your models and/or test image set are not on Google Cloud Storage buckets, you can upload them manually using the workbench GUI.)


2. To generate stats and visualize the results, run through the `visualize-predictions.ipynb` Jupyter notebook within your Vertex AI workbench.

(Note: If you have generated predictions elsewhere, create the `predictions` folder and put the CSV files into this folder, and make sure all files have the column headers, 'label' for the ground truth, and 'prediction' for the model-generated label.)