# Advanced Deep Learning for Robotics - SS21, Team 7

<!-- TODO: Fill the readme with project details. -->

## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.

## **Usage:**
    usage: main.py [-h] [--curiosity-model-path CURIOSITY_MODEL_PATH]
                   [--ppo-model-path PPO_MODEL_PATH] [--device DEVICE]

    optional arguments:
      -h, --help            show this help message and exit
      --curiosity-model-path CURIOSITY_MODEL_PATH
                            Path to the curiosity model file to be loaded/saved.
      --ppo-model-path PPO_MODEL_PATH
                            Path to the `RecurrentPPO` model file to be
                            loaded/saved. Note that it is a '.zip' file.
      --device DEVICE       String representation of the device to be used by
                            PyTorch.See https://pytorch.org/docs/stable/tensor_att
                            ributes.html#torch.torch.device for more details.
