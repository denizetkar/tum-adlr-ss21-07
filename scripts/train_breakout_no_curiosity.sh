#!/bin/bash

mkdir checkpoint
mkdir checkpoint/breakout_no_curiosity
mkdir tensorboard

python main.py \
  train \
  --atari \
  --ppo-model-path ./checkpoint/breakout_no_curiosity/ppo \
  --device cuda \
  --learning-rate 0.0001 \
  --tensorboard-log ./tensorboard/breakout_no_curiosity \
  --total-timesteps 200000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy CnnPolicy