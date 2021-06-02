#!/bin/bash

pip install pip-tools
pip-sync
wget http://www.atarimania.com/roms/Roms.rar
mkdir roms
sudo apt-get install unrar
unrar e Roms.rar ./roms/
python -m atari_py.import_roms ./roms/