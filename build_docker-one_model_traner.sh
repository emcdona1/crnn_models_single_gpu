#!/bin/bash
PROJECT_ID="handwriting-keras-tuner"
REGION="us-central1"
REPO="word-models"
IMAGE="iam-tuned"
TAG="dummy"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"

docker build . --file one_model_trainer/Dockerfile -t $IMAGE_URI
docker push $IMAGE_URI
