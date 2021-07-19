#!/bin/bash

mkdir -p checkpoints/breakout_curiosity_pure
mkdir -p tensorboard

python3 main.py \
  train \
  --atari \
  --curiosity-model-path "./checkpoints/breakout_curiosity_pure/curiosity" \
  --ppo-model-path "./checkpoints/breakout_curiosity_pure/ppo" \
  --device cuda \
  --learning-rate 0.0001 \
  --tensorboard-log "./tensorboard/breakout_curiosity_pure" \
  --curiosity-epochs 8 \
  --total-timesteps 200000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy CnnPolicy \
  --use-curiosity \
  --pure-curiosity-reward