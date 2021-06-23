#!/bin/bash

mkdir model
mkdir model/breakout_no_curiosity
mkdir tensorboard

python main.py \
  train \
  --ppo-model-path ./model/breakout_no_curiosity/ppo \
  --device cuda \
  --tensorboard-log ./tensorboard/breakout_no_curiosity \
  --total-timesteps 1000000

gsutil cp -r ./model gs://adlr-ss21-team7/breakout_no_curiosity
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/breakout_no_curiosity