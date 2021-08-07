#!/bin/bash

sudo mkdir -p /content/drive/MyDrive/experiment/checkpoints/mountaincar_curiosity
sudo mkdir -p /content/drive/MyDrive/experiment/tensorboard

python3 main.py \
  train \
  --curiosity-model-path "/content/drive/MyDrive/experiment/checkpoints/mountaincar_curiosity/curiosity" \
  --ppo-model-path "/content/drive/MyDrive/experiment/checkpoints/mountaincar_curiosity/ppo" \
  --device cuda \
  --alternate-train \
  --learning-rate 0.0003 \
  --min-batch-size 256 \
  --tensorboard-log "/content/drive/MyDrive/experiment/tensorboard/mountaincar_curiosity" \
  --ppo-epochs 4 \
  --curiosity-epochs 4 \
  --curiosity-reg-coef 0.001 \
  --total-timesteps 5000000 \
  --n-steps 32 \
  --n-envs 512 \
  --rnn-hidden-dim 256 \
  --policy MlpPolicy \
  --env "MountainCarContinuous-v0" \
  --use-curiosity