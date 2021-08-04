#!/bin/bash

mkdir -p checkpoints/pendulum_curiosity
mkdir -p tensorboard

python3 main.py \
  train \
  --curiosity-model-path "./checkpoints/pendulum_curiosity/curiosity" \
  --ppo-model-path "./checkpoints/pendulum_curiosity/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_curiosity" \
  --ppo-epochs 6 \
  --curiosity-epochs 4 \
  --total-timesteps 5000000 \
  --curiosity-reg-coef 0.001 \
  --n-steps 128 \
  --n-envs 64 \
  --policy MlpPolicy \
  --env "InvertedDoublePendulumBulletEnv-v0" \
  --pybullet-env \
  --use-curiosity