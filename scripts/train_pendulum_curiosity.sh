#!/bin/bash

mkdir -p checkpoints/pendulum_curiosity
mkdir -p tensorboard

python3 main.py \
  train \
  --curiosity-model-path "./checkpoints/pendulum_curiosity/curiosity" \
  --ppo-model-path "./checkpoints/pendulum_curiosity/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_curiosity" \
  --curiosity-epochs 8 \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 8 \
  --policy MlpPolicy \
  --env "InvertedDoublePendulumBulletEnv-v0" \
  --use-curiosity