#!/bin/bash

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-central1-a"
export INSTANCE_NAME="adlr-deep-learning-vm"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True" \
  --preemptible