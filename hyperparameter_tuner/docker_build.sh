#!/bin/bash
PROJECT_ID="handwriting-keras-tuner"
REGION="us-central1"
REPO="word-models"
IMAGE="iam-tuner"
TAG="hypertune"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"

docker build ./hyperparameter_tuner/ -t $IMAGE_URI
docker push $IMAGE_URI
