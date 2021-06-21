#!/bin/bash

mkdir checkpoint
mkdir checkpoint/pendulum_curiosity
mkdir tensorboard

python main.py \
  train \
  --curiosity-model-path "./checkpoint/pendulum_curiosity/curiosity.pth" \
  --ppo-model-path "./checkpoint/pendulum_curiosity/ppo.zip" \
  --device cuda \
  --tensorboard-log "./tensorboard/pendulum_curiosity" \
  --curiosity-epochs 8 \
  --total-timesteps 1000000 \
  --n-steps 2000 \
  --n-envs 8 \
  --rnn-hidden-dim 32 \
  --policy RnnPolicy \
  --env "InvertedDoublePendulum-v2" \
  --use-curiosity \
  --pure-curiosity-reward

gsutil cp -r ./checkpoint gs://adlr-ss21-team7/pendulum_curiosity
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/pendulum_curiosity