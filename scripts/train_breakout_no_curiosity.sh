#!/bin/bash

mkdir checkpoint
mkdir checkpoint/breakout_no_curiosity
mkdir tensorboard

python main.py \
  train \
  --atari \
  --ppo-model-path ./checkpoint/breakout_no_curiosity/ppo \
  --device cuda \
  --tensorboard-log ./tensorboard/breakout_no_curiosity \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 4 \
  --policy CnnPolicy