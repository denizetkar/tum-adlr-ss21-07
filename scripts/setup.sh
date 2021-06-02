#!/bin/bash

pip install gym
pip install stable-baselines3
pip install efficientnet_pytorch
pip install atari-py
pip install tensorboard
wget http://www.atarimania.com/roms/Roms.rar
mkdir roms
unar Roms.rar
python -m atari_py.import_roms ./Roms/