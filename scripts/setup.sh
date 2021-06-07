#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install gym
pip3 install stable-baselines3
pip3 install efficientnet_pytorch
pip3 install tensorboard

pip3 install atari-py
wget http://www.atarimania.com/roms/Roms.rar
mkdir roms
sudo apt-get install unar
unar Roms.rar
python -m atari_py.import_roms ./Roms/
