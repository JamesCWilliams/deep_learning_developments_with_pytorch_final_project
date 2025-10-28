#!/usr/bin/env bash

set -euo pipefail
PROJECT='rl-ga-variance'
TIMESTEPS=1000000
SEEDS=(1 2 3 4 5 6 7 8 9 10)
ENVS=('Hopper-v5' 'Ant-v5' 'HalfCheetah-v5' 'Pusher-v5')

run_algo () {
    local ALGO=$1 SCRIPT=$2 GROUP=$3
    for ENV in '${ENVS[@]}'; do
        for SEED in '${SEEDS[@]}'; do
            python -m cleanrl.${SCRIPT} \
            --env-id '${ENV}' --total-timesteps ${TIMESTEPS} \
            --seed ${SEED} --track \
            --wandb-project ${PROJECT} --wandb-group ${GROUP} \
            --wandb-name '${ALGO}_${ENV}_seed${SEED}'
        done
    done
}

run_algo 'ppo' 'ppo_continuous_action' 'ppo_baseline'
run_algo 'sac' 'sac_continuous_action' 'sac_baseline'
run_algo 'td3' 'td3_continuous_action' 'td3_baseline'
