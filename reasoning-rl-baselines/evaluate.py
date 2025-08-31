
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm

# Function to extract the final answer from a GSM8K response
def extract_final_answer(response):
    match = re.search(r"####\s*(\d+)", response)
    if match:
        return int(match.group(1))
    return None

def main(args):
    # Load the model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
    model = PeftModel.from_pretrained(base_model, args.adapter_model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the test dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Evaluation loop
    correct = 0
    total = 0
    for item in tqdm(dataset):
        question = item['question']
        prompt = tokenizer(question, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**prompt, max_new_tokens=256)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check for correctness
        predicted_answer = extract_final_answer(response)
        ground_truth_answer = extract_final_answer(item['answer'])

        if predicted_answer is not None and predicted_answer == ground_truth_answer:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy on {args.dataset_path}: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the base pretrained model")
    parser.add_argument("--adapter_model_path", type=str, required=True, help="Path to the trained adapter model (LoRA weights)")
    parser.add_argument("--dataset_path", type=str, default="gsm8k_test.jsonl", help="Path to the test dataset")
    
    args = parser.parse_args()
    main(args)
