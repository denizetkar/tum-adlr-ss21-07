#!/bin/bash

sudo mkdir -p /content/drive/MyDrive/experiment/checkpoints/breakout_curiosity
sudo mkdir -p /content/drive/MyDrive/experiment/tensorboard

python3 main.py \
  train \
  --atari \
  --curiosity-model-path "/content/drive/MyDrive/experiment/checkpoints/breakout_curiosity/curiosity" \
  --ppo-model-path "/content/drive/MyDrive/experiment/checkpoints/breakout_curiosity/ppo" \
  --device cuda \
  --alternate-train \
  --learning-rate 0.0001 \
  --min-batch-size 64 \
  --tensorboard-log "/content/drive/MyDrive/experiment/tensorboard/breakout_curiosity" \
  --ppo-epochs 4 \
  --curiosity-epochs 4 \
  --curiosity-reg-coef 0.001 \
  --total-timesteps 200000000 \
  --n-steps 16 \
  --n-envs 512 \
  --rnn-hidden-dim 256 \
  --policy CnnPolicy \
  --env "BreakoutNoFrameskip-v4" \
  --use-curiosity