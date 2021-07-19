mkdir -p checkpoints
mkdir -p tensorboard

rm -rf ./checkpoints/*
rm -rf ./tensorboard/*
rm -f nohup-*

nohup sh -c 'python3 main.py train --curiosity-model-path "./checkpoints/curiosity" --ppo-model-path "./checkpoints/recurrent-ppo-curiosity" --tensorboard-log "./tensorboard/MountainCarContinuous-v0_curiosity" --device cuda --curiosity-epochs 8 --total-timesteps 5000000 --n-steps 2000 --n-envs 8 --rnn-hidden-dim 32 --policy RnnPolicy --env "MountainCarContinuous-v0" --use-curiosity --pure-curiosity-reward' &> nohup-curiosity.out &
nohup sh -c 'python3 main.py train --ppo-model-path "./checkpoints/recurrent-ppo-no-curiosity" --tensorboard-log "./tensorboard/MountainCarContinuous-v0_no_curiosity" --device cuda --curiosity-epochs 8 --total-timesteps 5000000 --n-steps 2000 --n-envs 8 --rnn-hidden-dim 32 --policy RnnPolicy --env "MountainCarContinuous-v0"' &> nohup-no-curiosity.out &
tensorboard --logdir "./tensorboard" --port 6060
# python3 main.py play --ppo-model-path "./checkpoints/recurrent-ppo-curiosity.zip" --device "cuda" --n-envs 4 --env "MountainCarContinuous-v0"
# gsutil cp -r ./checkpoints gs://adlr-ss21-team7/MountainCarContinuous-v0
# gsutil cp -r ./tensorboard gs://adlr-ss21-team7/MountainCarContinuous-v0
