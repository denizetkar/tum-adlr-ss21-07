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
  --tensorboard-log "./tensorboard/breakout_curiosity" \
  --curiosity-epochs 8 \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 4 \
  --policy CnnPolicy \
  --use-curiosity