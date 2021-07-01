#!/bin/bash

mkdir checkpoint
mkdir checkpoint/pendulum_curiosity_pure
mkdir tensorboard

python main.py \
  train \
  --curiosity-model-path "./checkpoint/pendulum_curiosity_pure/curiosity" \
  --ppo-model-path "./checkpoint/pendulum_curiosity_pure/ppo" \
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