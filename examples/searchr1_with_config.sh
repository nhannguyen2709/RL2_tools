#!/bin/bash
# SearchR1 training script using YAML configuration
# Note: Ensure embedding server and search service are already running before executing this script
# You should firstly follow https://github.com/PeterGriffinJin/Search-R1 
# to download the index and corpus.

echo "Starting PPO training..."

# Run PPO training with configuration file
torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.ppo \
    --config-name=ppo_searchr1
    --actor.model_name=Qwen/Qwen3-0.6B