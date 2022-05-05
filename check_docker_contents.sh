#!/bin/bash
PROJECT_ID="handwriting-keras-tuner"
REGION="us-central1"
REPO="word-models"
IMAGE="iam-tuned"
TAG="run_55"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"

docker create --name="tmp_$$" $IMAGE_URI
docker export tmp_$$ | tar t > contents-$IMAGE_$TAG.tar
docker rm tmp_$$