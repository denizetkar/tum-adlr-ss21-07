#!/bin/bash

mkdir checkpoint
mkdir checkpoint/pendulum_curiosity_3m
mkdir tensorboard

python main.py \
  train \
  --curiosity-model-path "./checkpoint/pendulum_curiosity_3m/curiosity" \
  --ppo-model-path "./checkpoint/pendulum_curiosity_3m/ppo" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_curiosity_3m" \
  --curiosity-epochs 8 \
  --total-timesteps 3000000 \
  --n-steps 2000 \
  --n-envs 8 \
  --rnn-hidden-dim 32 \
  --policy RnnPolicy \
  --env "Acrobot-v1" \
  --use-curiosity \

gsutil cp -r ./checkpoint gs://adlr-ss21-team7/pendulum_curiosity_3m
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/pendulum_curiosity_3m