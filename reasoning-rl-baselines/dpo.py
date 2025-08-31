
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig
import re

def extract_final_answer(response):
    match = re.search(r"####\s*(\d+)", response)
    if match:
        return int(match.match(1))
    return None

def create_preference_dataset(dataset):
    """
    This function creates a preference dataset from the original GSM8K dataset.
    For DPO, we need pairs of (chosen, rejected) responses.
    This is a placeholder and might need to be adapted based on how you define 'chosen' vs 'rejected'.
    Here, we assume a response is 'chosen' if it's correct, and we would need to generate or find a 'rejected' response.
    For this example, we will just create a dummy rejected response.
    """
    preference_data = []
    for item in dataset:
        question = item['question']
        correct_answer_text = item['answer']
        
        # For simplicity, we'll use the ground truth as the chosen response
        # and a generic incorrect response as the rejected one.
        # A better approach would be to generate responses from a baseline model.
        chosen_response = f"{question}\n{correct_answer_text}"
        rejected_response = f"{question}\nI made a mistake in my calculation. The answer is 1234."

        preference_data.append({
            "prompt": question,
            "chosen": chosen_response,
            "rejected": rejected_response
        })
    
    return Dataset.from_list(preference_data)

def main(args):
    # Model and Tokenizer
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    original_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    preference_dataset = create_preference_dataset(original_dataset)

    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy="epoch",
    )

    # PEFT Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train the model
    dpo_trainer.train()

    # Save the model
    dpo_trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the pretrained model")
    parser.add_argument("--dataset_path", type=str, default="gsm8k_train.jsonl", help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, default="./dpo_gsm8k_model", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta for DPO")
    
    args = parser.parse_args()
    main(args)
