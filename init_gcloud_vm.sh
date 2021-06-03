#!/bin/bash

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-central1-b"
export INSTANCE_NAME="adlr-deep-learning-vm"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --boot-disk-size=200 \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"