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