# reasoning-rl-llm

Reasoning with Reinforcement Learning for LLMs.

This repository contains experiments with:
- `reasoning-rl-baselines`: PPO, GRPO, and DPO implementations.
- `reasoning-rl-curriculum`: Curriculum learning experiments.


create 3 python files and one requirements.txt - i need to experiment with reinforcement learning with PPO and GRPO using the trl library, and also compare the two RL implementations with DPO. the dataset i use is the gsm8k, already saved as gsm8k_train.jsonl and gsm8k_test.jsonl. in both PPO and GRPO scripts, i need to have the option of using correctness-only reward or a Llama 3.3 70b based evaluator to grade reasoning steps, in the latter case, make Reward = α*(final correctness) + β*(reasoning coherence score). I also need a 4th script to evaluate the trained models on the accuracy on GSM8K test set and do ablations of Correctness-only reward vs reasoning-aware reward. 
