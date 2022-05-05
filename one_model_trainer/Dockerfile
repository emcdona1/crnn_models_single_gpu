FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-7

WORKDIR /

# Installs hypertune library
# RUN pip install cloudml-hypertune

# Copies the trainer code to the docker image.
COPY trainer /trainer
COPY IAM_Words /IAM_Words

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]

