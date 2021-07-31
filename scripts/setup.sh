#!/bin/bash

sudo apt-get install -y python3-venv python3-dev || exit 1
python3 -m venv venv || exit 1
source ./venv/bin/activate || exit 1
pip3 install --upgrade pip || exit 1
# pip3 cache purge || exit 1
pip3 install opencv-python==4.5.3.56 || exit 1
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html || exit 1
pip3 install gym==0.18.3 || exit 1
pip3 install stable-baselines3==1.0 || exit 1
pip3 install efficientnet-pytorch==0.7.1 || exit 1
pip3 install tensorboard==2.5.0 || exit 1
pip3 install pybullet==3.1.7 || exit 1

pip3 install atari-py==0.2.9 || exit 1
wget http://www.atarimania.com/roms/Roms.rar || exit 1
mkdir roms || exit 1
sudo apt-get install -y unar || exit 1
unar Roms.rar || exit 1
python3 -m atari_py.import_roms ./Roms/ || exit 1

pip3 install jupyterlab || exit 1
nohup sh -c 'jupyter lab --no-browser' &> nohup.out &
