## Hyperparameter tuning

1. Update files as needed:

- Update the parameters in `hyperparameter_tuner\trainer_config.py` as needed (especially `num_epochs`, and confirm the metadata and folder names).

- Update the parameters in `hyperparameter_tuner\Dockerfile` as needed (especially the desired image folders).

- Update the variables in the `build_docker-hyperparameter_tuner.sh` to match your GCP project, selected region, Artifact Registry repo name, and desired image name and tag, as needed.

-----

2. Create new Docker image:

- Execute `build_docker-hyperparameter_tuner.sh` (e.g. `bash build_docker-hyperparameter_tuner.sh`).

-----

3. Start the hyperparameter tuning in the UI (Leave the workbench open so you can reference it):

- Back in the Vertex AI UI, navigate to Training, then click "Create."

1. Training Method:
    * Dataset: No managed dataset
    * Model training method: Custom trainin (advanced)
2. Model details:
    * Train new model: give it a name
3. Training container:
    * Click "Custom container."
    * Under Container Image, browse to your newly created image in the Artifact Registry.
    * Set the Model Output Directory to somewhere within a Storage bucket.
4. Hyperparameter tuning:
    * Check the box to enable it!
    * Tedious - for each hyperparameter (see arguments in `task.py`) enter the name (must match exactly), type, and desired values.
    * Specify the evaluation metric as named in Keras, and the goal (e.g. `val_loss` and Minimize).
    * Specify the max number of trials to conduct (longer, more expensive), and the number of parallel trials (faster, but search won't be conducted as efficiently).
5. Compute and pricing:
    * Specify at least one machine (n1-standard-4 is plenty for IAM dataset).
    * Recommended to select an accelerator (GPU).  (Training on a Tesla-V100 GPU for 60 trials, 3 parallel, 15 epochs each, took ~30 hours and cost ~$250.)
6. Click Start Training.
