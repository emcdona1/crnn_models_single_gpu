# Training a single model

0. Optional: Create nonoverlapping training/testing sets when training the actual models.

-

-----

1. Update files as needed:

- Update the parameters in `one_model_trainer\trainer_config.py` as needed (especially `num_epochs`, and confirm the metadata and folder names).

- Update the parameters in `one_model_trainer\Dockerfile` as needed (especially the desired image folders).

- Update the variables in the `build_docker-one_model_trainer.sh` to match your GCP project, selected region, Artifact Registry repo name, and desired image name and tag, as needed.

-----

2. Create new Docker image:

- Execute `build_docker-one_model_trainer.sh` (e.g. `bash build_docker-one_model_trainer.sh`).

-----

3. Start the model training, using the GCP UI (Leave the workbench open so you can reference it):

- Back in the Vertex AI UI, navigate to Training, then click "Create."

1. Training Method:
    * Dataset: No managed dataset
    * Model training method: Custom trainin (advanced)
2. Model details:
    * Train new model: give it a name
3. Training container:
    * Click "Custom container."
    * Under Container Image, browse to your newly created image in the Artifact Registry.
    * Set the Model Output Directory to somewhere within a Storage bucket, e.g. `gs://iam-model-staging/<model_id>\`, by creating a new folder called `<model_id>`.
4. Hyperparameter tuning:
    * Click continue.
5. Compute and pricing:
    * Specify at least one machine (n1-standard-4 is plenty for IAM dataset).
    * Highly recommended to select an accelerator (GPU).
6. Prediction container:
    * You can easily create one later, as needed.  The trained model & history information will be saved in the output directory you specified in the Training container tab.
7. Click Start Training.