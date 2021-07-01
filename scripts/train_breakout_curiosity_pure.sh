#!/bin/bash

mkdir checkpoint
mkdir checkpoint/breakout_curiosity_pure
mkdir tensorboard

python main.py \
  train \
  --atari \
  --curiosity-model-path "./checkpoint/breakout_curiosity_pure/curiosity" \
  --ppo-model-path "./checkpoint/breakout_curiosity_pure/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/breakout_curiosity_pure" \
  --curiosity-epochs 8 \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 4 \
  --policy CnnPolicy \
  --use-curiosity \
  --pure-curiosity-reward