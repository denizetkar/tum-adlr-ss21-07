#!/bin/bash

mkdir -p checkpoints/pendulum_no_curiosity
mkdir -p tensorboard

python3 main.py \
  train \
  --ppo-model-path "./checkpoints/pendulum_no_curiosity/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_no_curiosity" \
  --ppo-epochs 6 \
  --total-timesteps 5000000 \
  --n-steps 128 \
  --n-envs 64 \
  --policy MlpPolicy \
  --env "InvertedDoublePendulumBulletEnv-v0" \
  --pybullet-env