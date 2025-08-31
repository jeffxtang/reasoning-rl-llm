
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig
import re
from tqdm import tqdm

# Function to extract the final answer from a GSM8K response
def extract_final_answer(response):
    match = re.search(r"####\s*(\d+)", response)
    if match:
        return int(match.group(1))
    return None

import together
import os

# Configure the Together.ai client
# Make sure to set the TOGETHER_API_KEY environment variable
client = together.Together(api_key=os.environ.get("TOGETHER_API_KEY"))

def get_reasoning_coherence_score(reasoning: str) -> torch.Tensor:
    """
    Evaluates the coherence of reasoning steps using a Llama 3 70b model on Together.ai.
    """
    prompt = f"""
    You are an expert evaluator of mathematical reasoning. Your task is to assess the coherence and logical flow of the provided reasoning steps for a math problem.
    The reasoning should be clear, correct, and easy to follow.

    Provided Reasoning:
    ---
    {reasoning}
    ---

    Please evaluate the coherence of this reasoning on a scale from 0.0 to 1.0, where 0.0 is completely incoherent and 1.0 is perfectly coherent.
    Provide only the score in the format "SCORE: <score>".
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",  # Using a Llama 3 70b model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        content = response.choices[0].message.content
        match = re.search(r"SCORE:\s*([0-9.]+)", content)
        if match:
            score = float(match.group(1))
            return torch.tensor(score)
        else:
            print(f"Could not parse score from response: {content}")
            return torch.tensor(0.5)  # Return neutral score on parsing failure
    except Exception as e:
        print(f"Error calling Together.ai API: {e}")
        return torch.tensor(0.5)  # Return neutral score on API error

def main(args):
    # PPO Configuration
    ppo_config = PPOConfig(
        p_coef=args.p_coef,
        init_kl_coef=args.init_kl_coef,
        target=args.target,
        horizon=args.horizon,
        gamma=args.gamma,
        lam=args.lam,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        batch_size=args.batch_size,
        forward_batch_size=args.forward_batch_size,
        adap_kl_ctrl=args.adap_kl_ctrl,
    )

    # Model and Tokenizer
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # PEFT Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0]),
        peft_config=lora_config,
    )

    # Training Loop
    for epoch in tqdm(range(args.epochs), "epoch"):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = [tokenizer.encode(q, return_tensors="pt").to(model.device) for q in batch["query"]]
            
            # Get response from the model
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False)
            batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            # Calculate reward
            rewards = []
            for i in range(len(batch['response'])):
                final_answer = extract_final_answer(batch['response'][i])
                correct_answer = extract_final_answer(batch['answer'][i])
                
                correctness = 1.0 if final_answer == correct_answer else 0.0
                
                if args.reward_type == 'correctness_only':
                    reward = torch.tensor(correctness)
                elif args.reward_type == 'reasoning_aware':
                    reasoning_score = get_reasoning_coherence_score(batch['response'][i])
                    reward = args.alpha * correctness + args.beta * reasoning_score
                else:
                    raise ValueError("Invalid reward type specified")
                
                rewards.append(reward)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    # Save the model
    ppo_trainer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the pretrained model")
    parser.add_argument("--dataset_path", type=str, default="gsm8k_train.jsonl", help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, default="./ppo_gsm8k_model", help="Directory to save the trained model")
    parser.add_argument("--reward_type", type=str, default="correctness_only", choices=["correctness_only", "reasoning_aware"], help="Type of reward to use")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for correctness in reasoning-aware reward")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for reasoning coherence in reasoning-aware reward")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    # PPOConfig arguments
    parser.add_argument("--p_coef", type=float, default=0.1)
    parser.add_argument("--init_kl_coef", type=float, default=0.2)
    parser.add_argument("--target", type=float, default=6.0)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--forward_batch_size", type=int, default=16)
    parser.add_argument("--adap_kl_ctrl", type=bool, default=True)
    
    args = parser.parse_args()
    main(args)
