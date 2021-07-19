#!/bin/bash

mkdir checkpoint
mkdir checkpoint/breakout_curiosity
mkdir tensorboard

python main.py \
  train \
  --atari \
  --curiosity-model-path "./checkpoint/breakout_curiosity/curiosity" \
  --ppo-model-path "./checkpoint/breakout_curiosity/ppo" \
  --device cuda \
  --learning-rate 0.0001 \
  --tensorboard-log "./tensorboard/breakout_curiosity" \
  --curiosity-epochs 8 \
  --total-timesteps 200000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy CnnPolicy \
  --use-curiosity