# TODO: Update the COPY commands as needed for your experiment

FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-7

WORKDIR /

# Copies the trainer code to the docker image.
COPY utilities /utilities
COPY setup.cfg /setup.cfg
COPY one_model_trainer /trainer
COPY images/IAM_Words_train /IAM_Words

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]

