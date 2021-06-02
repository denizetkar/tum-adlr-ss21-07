#!/bin/bash

mkdir models
mkdir models/breakout_curiosity
mkdir models/breakout_curiosity/curiosity
mkdir models/breakout_curiosity/ppo
mkdir tensorboard

python main.py \
  --ppo-model-path ./models/breakout_curiosity/ppo \
  --device cuda \
  --tensorboard-log ./tensorboard/breakout_curiosity \
  --total-timesteps 1000000

gsutil cp -r ./models gs://adlr-ss21-team7/breakout_no_curiosity
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/breakout_no_curiosity