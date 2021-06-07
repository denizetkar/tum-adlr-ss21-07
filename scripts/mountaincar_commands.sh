mkdir -p checkpoints
mkdir -p tensorboard

rm -rf ./checkpoints/*
rm -rf ./tensorboard/*
rm -f nohup-*

nohup sh -c 'python main.py train --curiosity-model-path "./checkpoints/curiosity.model" --ppo-model-path "./checkpoints/recurrent-ppo-curiosity.zip" --tensorboard-log "./tensorboard/MountainCarContinuous-v0_curiosity" --device cuda --total-timesteps 5000000 --rnn-hidden-dim 32 --policy RnnPolicy --env "MountainCarContinuous-v0" --use-curiosity --pure-curiosity-reward' &> nohup-curiosity.out &
nohup sh -c 'python main.py train --curiosity-model-path "./checkpoints/no-curiosity.model" --ppo-model-path "./checkpoints/recurrent-ppo-no-curiosity.zip" --tensorboard-log "./tensorboard/MountainCarContinuous-v0_no_curiosity" --device cuda --total-timesteps 5000000 --rnn-hidden-dim 32 --policy RnnPolicy --env "MountainCarContinuous-v0"' &> nohup-no-curiosity.out &
tensorboard --logdir "./tensorboard" --port 6060

# gsutil cp -r ./checkpoints gs://adlr-ss21-team7/MountainCarContinuous-v0
# gsutil cp -r ./tensorboard gs://adlr-ss21-team7/MountainCarContinuous-v0
