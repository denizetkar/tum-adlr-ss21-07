#!/bin/bash

mkdir -p checkpoints/breakout_curiosity
mkdir -p tensorboard

python3 main.py \
  train \
  --atari \
  --curiosity-model-path "./checkpoints/breakout_curiosity/curiosity" \
  --ppo-model-path "./checkpoints/breakout_curiosity/ppo" \
  --device cuda \
  --learning-rate 0.0001 \
  --tensorboard-log "./tensorboard/breakout_curiosity" \
  --ppo-epochs 4 \
  --curiosity-epochs 6 \
  --total-timesteps 200000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy CnnPolicy \
  --env "BreakoutNoFrameskip-v4" \
  --use-curiosity