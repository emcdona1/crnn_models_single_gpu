# Training & Tuning CRNN Handwriting Models

The code in this repository is designed to perform hyperparameter tuning, model training,
transfer learning, and performance comparison with handwritten word datasets.  The code is designed
to use one GPU core at a time, but can be run on CPU-only machines.

This repo is built on Python 3.7, and requires the additional packages listed in `requirements.txt`.


## Suggested Resources
* For running on a server with Tensorflow-compatible GPU(s), make sure all required NVIDIA/CUDA 
software is installed.  (i.e. execute `tf.config.list_physical_devices('GPU')` in Python, and confirm
it returns at least one GPU.)
* Suggestions for how to access the server via command-line and SFTP are in 
[Felix Grewe's Linux Cookbook](https://felixgrewe.github.io/linux_cookbook/login_server.html).

-----

## Workflow

Note: On a multi-GPU system, all of these scripts below select the one GPU with the most available
capacity.  When no GPUs are detected, this will run on the CPU (e.g. for local testing).

-----

### 0. Setup

1. Install the necessary requirements in your environment, using `requirements.txt`.
2. Modify the fields in `setup.cfg` to match your files.  Most of these fields are self-evident, but a few
are worthy of explanation:
   - METADATA_IMAGE_COLUMN = the title/header for the column in the metadata file that contains the image name.
   - METADATA_TRANSCRIPTION_COLUMN - the title/header for the column in the metadata file that contains 
      the actual transcribde text (ground truth label).
   - [tune] [train] and [test] contain parameters used for (respectively) hyperparameter tuning,
      training _and_ transfer learning, and testing.  The [train] parameters can be overwritten using arguments (details below).


### 1. Hyperparameter Tuning

Note: This is executed with a Naive Bayes search.  A full grid search in TensorBoard is available in 
the file `task-hparam_grid_search.py`.

1. Under the import statements, modify the search values as desired.  Currently, you can modify:
   - Batch size
   - Convolutional kernel size (square)
   - Number of fully connected units in the first dense layer
   - Dropout after the convolutional layers
   - Number of units in the first LSTM layer.
   - Number of units in the second LSTM layer.
   - Learning rate.
2. Confirm the `setup.cfg` parameters under [project] and [tune] are set as desired.
3. Execute `task.py` from the main directory, e.g.:

`python hyperparameter_tuner/task.py setup.cfg`

4. The model saves run results in the `logs/hparam_tuning_grid` folder, and saves a pickle file 
   (`tuner_search_results.pkl`) for viewing the search results.  
   (**N.B.** This folder gets cleared out at the start of each run!)
5. After tuning, the program prints out the best hyperparameters.  You can view the results by executing
   the `bayes_search_results.ipynb` Jupyter notebook, and loading the pickled file.

-----

### 2. Training A Model

1. In the `setup.cfg` file, confirm the [project] and [train] parameters are set as desired, based on the identified
   hyperparameters and training image set.
3. Execute `task.py` from the main directory, e.g.:

`python one_model_trainer/task.py setup.cfg example_model_name -[optional flags]`

4. You have the option to pass in hyperparameters using the `setup.cfg` file, or to override these parameter values
   by using the optional flags.  Run `python one_model_trainer/task.py setup.cfg example_model_name -h`
   for full details.
5. The program saves 2 files to the `saved_models` folder: a `-full_model.h5` file 
   which contains the complete model (including CTC layer), for further training, and a second model
   (CTC layer removed) which can be used for predictions.

-----

### 3. Transfer Learning for Fine-tuning Model(s)

1. In the `setup.cfg` file, confirm the project and 
   training parameters are set as desired.
2. At the top of the program, you can change `RETRAINING_EPOCHS` and `FINE_TUNING_EPOCHS`
   as desired.
3. Execute `task.py` from the main directory, e.g.:

`python transfer_learning/task.py setup.cfg saved_models/prediction_model.h5 example_model_name`

4. The program saves 4 files to the `saved_models` folder: two `-retrained.h5`
   and two `-fine_tuned.h5` models (see "Training a Tuned Model" for explanation of the two models).

-----

### 4. Generating Predictions on the Test Set

1. In the `setup.cfg` file, confirm the image set parameters are set as desired.
2. Execute `task.py` from the main directory, with 1+ model files, e.g.:

`python test_set_predictions/task.py setup.cfg saved_models/prediction_model.h5 saved_models/prediction_model_2.h5 ...`

3. For each model file provided, the program saves 1 CSV file to the `test_set_predictions\predictions` folder,
   containing the ground truth texts and the prediction text.

-----

### 5. Visualizing Performance

Open `test_set_predictions/visulaize_predictions.ipynb` in a Jupyter environment.
This notebook loads all prediction CSV files in the `predictions` folder, and generates
statistics, summary statistics, and comparison graphs.

-----

## Contributors and licensing
This code has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1), *Field Museum*). 

This code was developed under the guidance of [Dr. Matt von Konrat](https://www.fieldmuseum.org/about/staff/profile/16) (Field Museum), and [Dr. Rick Ree](https://www.fieldmuseum.org/about/staff/profile/36) (*Field Museum*).

This project was made possible thanks to [the Grainger Bioinformatics Center](https://www.fieldmuseum.org/science/labs/grainger-bioinformatics-center) at the Field Museum.

Please contact Dr. von Konrat for licensing inquiries.
