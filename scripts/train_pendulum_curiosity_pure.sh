#!/bin/bash

mkdir -p checkpoints/pendulum_curiosity_pure
mkdir -p tensorboard

python main.py \
  train \
  --curiosity-model-path "./checkpoints/pendulum_curiosity_pure/curiosity" \
  --ppo-model-path "./checkpoints/pendulum_curiosity_pure/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_curiosity_pure" \
  --curiosity-epochs 8 \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 8 \
  --policy MlpPolicy \
  --env "InvertedDoublePendulumBulletEnv-v0" \
  --use-curiosity \
  --pure-curiosity-reward