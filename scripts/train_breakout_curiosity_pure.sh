#!/bin/bash

mkdir model
mkdir model/breakout_curiosity
mkdir tensorboard

python main.py \
  train \
  --atari \
  --use-curiosity \
  --curiosity-model-path ./model/breakout_curiosity/curiosity \
  --ppo-model-path ./model/breakout_curiosity/ppo.pth \
  --device cuda \
  --tensorboard-log ./tensorboard/breakout_curiosity \
  --total-timesteps 1000000

gsutil cp -r ./model gs://adlr-ss21-team7/breakout_curiosity
gsutil cp -r ./tensorboard gs://adlr-ss21-team7/breakout_curiosity