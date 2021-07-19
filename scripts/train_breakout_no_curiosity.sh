#!/bin/bash

mkdir -p checkpoints/breakout_no_curiosity
mkdir -p tensorboard

python3 main.py \
  train \
  --atari \
  --ppo-model-path ./checkpoints/breakout_no_curiosity/ppo \
  --device cuda \
  --learning-rate 0.0001 \
  --tensorboard-log ./tensorboard/breakout_no_curiosity \
  --total-timesteps 200000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy CnnPolicy