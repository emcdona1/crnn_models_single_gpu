# Training & Tuning CRNN Handwriting Models on a local GPU server

The code in this repository has been used to perform hyperparameter tuning, model training, and (not yet) model deployment within Google Cloud's infrastructure.

This code is built on Python 3.7.13.

## Pre-Requirements
* Access to a server with Tensorflow-compatible GPUs, with all required NVIDIA/CUDA software installed. 
(Executing `tf.config.list_physical_devices('GPU')` should return at least one GPU.)
* Command-line access to the server (e.g. KiTTY, PuTTY).
* FTP access to the server (e.g. WinSCP).

-----

## Suggested Workflow

Note: On a multi-GPU system, all of these scripts below select the one GPU with the most available
capacity.  When no GPUs are detected, this will run on the CPU (e.g. for local testing).

-----

### 1. Hyperparameter Tuning

Note: This is currently a full grid search, executed using TensorBoard.

1. In the bottom `__main__` function of the code, modify the search values as desired.  Currently you can modify:
   - Batch size
   - Convolutional kernel size (square)
   - Number of fully connected units in the first dense layer
   - Dropout after the convolutional layers
   - Number of units in the first LSTM layer.
   - Learning rate.
2. In the `setup.cfg` file, confirm the project parameters and 
   NUM_EPOCHS and SEED (under 'train') are set as desired.
3. Execute `task.py` from the main directory, e.g.:

`python hyperparameter_tuner/task.py setup.cfg`

4. The model saves run results in the `logs/hparam_tuning_grid` folder. 
   (Note that this folder gets cleared out at the start of each run.)
5. To view the results, you can execute `tensorboard --logdir logs/hparam_tuning_grid` in the console to view TensorBoard.

-----

### 2. Training A Tuned Model

1. In the `setup.cfg` file, confirm the project and 
   training parameters are set as desired.
2. At the top of the `main()` method, set the parameters which were selected during hyperparameter tuning.
3. Execute `task.py` from the main directory, e.g.:

`python one_model_trainer/task.py setup.cfg example_model_name`

4. The program saves 2 files to the `saved_models` folder: a `-full_model.h5` file 
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
