# Training & Tuning CRNN Handwriting Models on Google Cloud

The code in this repository has been used to perform hyperparameter tuning, model training, and (not yet) model deployment within Google Cloud's infrastructure.

## Google Cloud Platform account requirements
* Billable Google Cloud account (hyperparameter tuning, especially, is expensive)
* The following APIs enabled:
	* Cloud Storage
	* Vertex AI
	* Artifact Registry
	* (optional) GPU quota of >= 1 ([helpful tutorial](https://stackoverflow.com/questions/53415180/gcp-error-quota-gpus-all-regions-exceeded-limit-0-0-globally))

-----

## Workflows

### 1. Hyperparameter tuning

GCP enviorment setup:
1. In GCP Artifact Registry, create a repository for your Docker image (make sure it's in the same Region as everything else), then navigate inside the new repo and click "Setup Instructions" near the top.
2. Copy the "Configure Docker" command  (Something similar to `gcloud auth configure-docker us-central1-docker.pkg.dev`).
3. In GCP Vertex AI, navigate to Workbench --> Managed Notebooks, and click New Notebook near the top.
4. Name your new notebook, make sure it's in the same Region as everything else, change Permission to "Service account," and under "Advanced" be sure to check "Enable terminal" (and anything else you want to change).
5. After a few minutes, you can click "Open Jupyer Lab" for the new notebook.  Clone this repo into your notebook.

Hypertuner setup:
1. Copy your image set + metadata file into the workspace. (e.g. `gsutil -m cp -r gs://{storage_bucket_name}/IAM_Words`).
2. Navigate to the `hyperparameter_tuner` folder.
3. Update the parameters in `trainer_config.py` as needed (especially `num_epochs`, and check the metadata and folder names).
4. Update the variables in the `docker_build.sh` to match your GCP project, region, artifact repo name, and desired image name, as needed.
5. Open a Terminal window and `cd` to the main repo folder (e.g. `cd  handwriting_models_on_vertex_ai`).
6. Execute `docker_build.sh` (e.g. `bash hyperparameter_tuner\docker_build.sh`).
7. Leave the workbench open so you can reference it in step 4 below.

Starting the tuning:
* In Vertex AI --> Training --> Click "Create."
* 1. Training Method:
  * Dataset: No managed dataset
  * Model training method: Custom trainin (advanced)
* 2. Model details:
  * Train new model: give it a name
* 3. Training container:
  * Click "Custom container."
  * Under Container Image, browse to your newly created image in the Artifact Registry.
  * Set the Model Output Directory to somewhere within a Storage bucket.
* 4. Hyperparameter tuning:
  * Check the box to enable it!
  * Tedious - for each hyperparameter (see arguments in `task.py`) enter the name (must match exactly), type, and desired values.
  * Specify the evaluation metric as named in Keras, and the goal (e.g. `val_loss` and Minimize).
  * Specify the max number of trials to conduct (longer, more expensive), and the number of parallel trials (faster, but search won't be conducted as efficiently).
* 5. Compute and pricing:
  * Specify at least one machine (n1-standard-4 is plenty for IAM dataset).
  * Recommended to select an accelerator (GPU).  (Training on a Tesla-V100 GPU for 60 trials, 3 parallel, 15 epochs each, took ~30 hours and cost ~$250.)
* Click Start Training.

-----

2. Training one specific model
(tk)
