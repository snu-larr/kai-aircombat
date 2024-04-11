#!/bin/sh
env="SingleControl"
scenario="1/sam"
algo="ppo"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 \

python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 1 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 600 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 10 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8