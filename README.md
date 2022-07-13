# Training & Tuning CRNN Handwriting Models on Google Cloud

The code in this repository has been used to perform hyperparameter tuning, model training, and (not yet) model deployment within Google Cloud's infrastructure.

This code is built on Python 3.7.12.

## Google Cloud Platform account requirements
* Billable Google Cloud account (hyperparameter tuning, especially, is expensive)
* The following APIs enabled:
	* Cloud Storage
	* Vertex AI
	* Artifact Registry
	* (optional) GPU quota of >= 1 ([helpful tutorial](https://stackoverflow.com/questions/53415180/gcp-error-quota-gpus-all-regions-exceeded-limit-0-0-globally))

-----

## Google Cloud Platform environment setup

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

-----

See modules for specific steps after setup.

-----

## Contributors and licensing
This code has been developed by Beth McDonald ([emcdona1](https://github.com/emcdona1), *Field Museum*). 

This code was developed under the guidance of [Dr. Matt von Konrat](https://www.fieldmuseum.org/about/staff/profile/16) (Field Museum), and [Dr. Rick Ree](https://www.fieldmuseum.org/about/staff/profile/36) (*Field Museum*).

This project was made possible thanks to [the Grainger Bioinformatics Center](https://www.fieldmuseum.org/science/labs/grainger-bioinformatics-center) at the Field Museum.

Please contact Dr. von Konrat for licensing inquiries.
