#!/bin/bash

mkdir checkpoint
mkdir checkpoint/pendulum_no_curiosity
mkdir tensorboard

python main.py \
  train \
  --ppo-model-path "./checkpoint/pendulum_no_curiosity/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_no_curiosity" \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 8 \
  --rnn-hidden-dim 32 \
  --policy RnnPolicy \
  --env "Acrobot-v1"

gsutil cp -r ./checkpoint gs://adlr-ss21-team7/pendulum_no_curiosity
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/pendulum_no_curiosity