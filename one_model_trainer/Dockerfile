# TODO: Update the COPY commands as needed for your experiment

FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-7

WORKDIR /

# Copies the trainer code to the docker image.
COPY one_model_trainer /trainer
# COPY IAM_Words /IAM_Words
COPY IAM_Words_train /IAM_Words 
# COPY IAM_Words_test /IAM_Words

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]

