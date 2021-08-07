#!/bin/bash

sudo mkdir -p /content/drive/MyDrive/experiment/checkpoints/breakout_no_curiosity
sudo mkdir -p /content/drive/MyDrive/experiment/tensorboard

python3 main.py \
  train \
  --atari \
  --ppo-model-path "/content/drive/MyDrive/experiment/checkpoints/breakout_no_curiosity/ppo" \
  --device cuda \
  --learning-rate 0.0001 \
  --min-batch-size 64 \
  --tensorboard-log "/content/drive/MyDrive/experiment/tensorboard/breakout_no_curiosity" \
  --ppo-epochs 4 \
  --total-timesteps 200000000 \
  --n-steps 16 \
  --n-envs 512 \
  --rnn-hidden-dim 256 \
  --policy CnnPolicy \
  --env "BreakoutNoFrameskip-v4"