from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import tune


tune.run(PPOTrainer, config={
    "env": "CartPole-v0",
    "framework": "torch",
    "evaluation_num_workers": 1,
    "evaluation_interval": 10,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "monitor": True
    }}, local_dir="runs", stop={"training_iteration": 100})
