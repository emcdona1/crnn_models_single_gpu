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

### 0. Google Cloud Platform environment setup

1. In Cloud Storage, upload your image set (create a bucket if necessary, e.g. `gs://fmnh_datasets/`).  Do this using the web UI, or using gsutil on your local machine (e.g. `gsutil -m cp -r <source_folder> gs://fmnh_datasets/<dataset_name>`). 
3. In Vertex AI, navigate to Workbench --> Managed Notebooks, and click New Notebook near the top.
4. Name your new notebook, make sure it's in the same Region as everything else, change Permission to "Service account," and under "Advanced" be sure to check "Enable terminal" (and anything else you want to change).
5. In the meantime, go to Artifact Registry and create a repository for your Docker image, making sure it's in the same Region as everything else. Navigate inside this new repo and click "Setup Instructions" near the top.
6. Copy the "Configure Docker" command.  (Something similar to `gcloud auth configure-docker us-central1-docker.pkg.dev`)
7. Return to Vertex AI.  After a few minutes, you can click "Open Jupyer Lab" for the new notebook.
8. Clone this repo into your notebook.  (using a Terminal window, e.g. `git clone https://github.com/emcdona1/handwriting_models_on_vertex_ai/`)
2. Within your managed notebook, in a Terminal window, navigate to the repo folder. (e.g. 'cd handwriting_models_on_vertex_ai`).
3. In the same Terminal window, paste and run your configure Docker command.
9. Copy the image set (including metadata) into the workspace. (using a Terminal window, e.g. `gsutil -m cp -r gs://fmnh_datasets/IAM_Words ./`)


### 1. Hyperparameter tuning
1. Update the parameters in `hyperparameter_tuner\trainer_config.py` as needed (especially `num_epochs`, and confirm the metadata and folder names).
2. Update the parameters in `hyperparameter_tuner\Dockerfile` as needed (especially the desired image folders).
4. Update the variables in the `build_docker-hyperparameter_tuner.sh` to match your GCP project, selected region, Artifact Registry repo name, and desired image name and tag, as needed.
6. Execute `build_docker-hyperparameter_tuner.sh` (e.g. `bash build_docker-hyperparameter_tuner.sh`).
7. Leave the workbench open so you can reference it below.
8. Back in the Vertex AI UI, navigate to Training, then click "Create."
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

-----

2. Training one specific model
(tk)
