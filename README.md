# Advanced Deep Learning for Robotics - SS21, Team 7

<!-- TODO: Fill the readme with project details. -->

## **Installing Dependencies:**
### Option 1: Setup script
* If you have a Linux operating system run `./scripts/setup.sh` from the project root directory.
### Option 2: Manual installation
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.
* Download Roms.rar from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/roms/Roms.rar) and extract the .rar file.
* Run `python -m atari_py.import_roms <path to folder that holds the content of Roms.rar>`

## **Usage:**
### Option 1: Use `main.py` to train
    usage: main.py [-h] {train,play} ...

    optional arguments:
      -h, --help    show this help message and exit

    Subcommand:
      {train,play}
        train       Subcommand for training the PPO and curiosity models.
        play        Subcommand for getting the PPO model to play in the
                    environment for which it was trained.

#### `train` subcommand usage
    usage: main.py train [-h] [--curiosity-model-path CURIOSITY_MODEL_PATH]
                         --ppo-model-path PPO_MODEL_PATH --tensorboard-log
                         TENSORBOARD_LOG [--device DEVICE]
                         [--curiosity-epochs CURIOSITY_EPOCHS]
                         [--total-timesteps TOTAL_TIMESTEPS] [--n-steps N_STEPS]
                         [--n-envs N_ENVS] [--rnn-hidden-dim RNN_HIDDEN_DIM]
                         [--policy {RnnPolicy,CnnRnnPolicy}] [--env ENV]
                         [--partially-observable] [--use-curiosity]
                         [--pure-curiosity-reward]

    optional arguments:
      -h, --help            show this help message and exit
      --curiosity-model-path CURIOSITY_MODEL_PATH
                            Path to the curiosity model file to be loaded/saved.
      --ppo-model-path PPO_MODEL_PATH
                            Path to the `RecurrentPPO` model file to be
                            loaded/saved. Note that it is a '.zip' file.
      --tensorboard-log TENSORBOARD_LOG
                            Path to the directory for saving the tensorboard logs.
                            Directory will be created if it does not exist.
      --device DEVICE       String representation of the device to be used by
                            PyTorch.See https://pytorch.org/docs/stable/tensor_att
                            ributes.html#torch.torch.device for more details.
      --curiosity-epochs CURIOSITY_EPOCHS
                            Number of epochs to train the curiosity models per
                            'collect_rollout'.
      --total-timesteps TOTAL_TIMESTEPS
                            Total number of timestamps for training
      --n-steps N_STEPS     Maximum number of timesteps per rollout per
                            environment
      --n-envs N_ENVS       Number of environments for data collection
      --rnn-hidden-dim RNN_HIDDEN_DIM
                            Hidden dimension size for RNNs
      --policy {RnnPolicy,CnnRnnPolicy}
                            Type of the policy network
      --env ENV             String representation of the gym environment
      --partially-observable
                            Flag for informing the model to use RNNs due to
                            partial observability of the RL problem.
      --use-curiosity       Flag for using curiosity in the training
      --pure-curiosity-reward
                            Flag for telling to use pure curiosity rewards instead
                            of mixed curiosity reward (extrinsic + coef *
                            curiosity).

#### `play` subcommand usage
    usage: main.py play [-h] --ppo-model-path PPO_MODEL_PATH [--device DEVICE]
                        [--n-envs N_ENVS] [--env ENV]

    optional arguments:
      -h, --help            show this help message and exit
      --ppo-model-path PPO_MODEL_PATH
                            Path to the `RecurrentPPO` model file to be
                            loaded/saved. Note that it is a '.zip' file.
      --device DEVICE       String representation of the device to be used by
                            PyTorch.See https://pytorch.org/docs/stable/tensor_att
                            ributes.html#torch.torch.device for more details.
      --n-envs N_ENVS       Number of environments for data collection
      --env ENV             String representation of the gym environment

### Option 2: Use pre-defined scripts to train. More scripts will be added soon.
Run one the following scripts **from the project root**

#### Training the Atari Game 'Breakout' with actor-critic CnnRnnPolicy
```./scripts/train_breakout_no_curiosity.sh```

#### Training the Atari Game 'Breakout' with curiosity-driven exploration and actor-critic CnnRnnPolicy
```./scripts/train_breakout_curiosity.sh```