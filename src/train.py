import argparse
import os
import time
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from datasets import load_dataset
from src.config.config import LoraConfig, TrainingConfig
from src.datasets.dialogsum import subsample_dataset, tokenize_function
from src.trainers.trainer import BaseTrainer
from src.utils.utils import load_prompt_template, print_trainable_model_params


def main(args):
    # prepare output dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        args.output_dir, f"{args.training_method}-run-{timestamp}"
    )

    # load configs
    train_config = TrainingConfig(
        output_dir=output_dir
    )
    if args.training_method == "peft":
        lora_config = LoraConfig(target_modules=["q","v"])
    else:
        lora_config = None

    # Load dataset and prompt
    dataset = load_dataset(args.dataset_name)
    prompt_template = load_prompt_template(args.prompt_template)

    # Load Trainer
    trainer = BaseTrainer(
        model_name= args.model,
        train_config= train_config,
        lora_config = lora_config,
        precision = args.precision,
    )

    # Apply tokenization to the dataset
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, trainer.tokenizer, prompt_template),
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary']
    )
    print("Tokenization complete!")

    # Set the format to PyTorch tensors
    tokenized_datasets = tokenized_datasets.with_format("torch")
    
    # Subsample dataset if specified
    if args.subsample:
        tokenized_datasets = subsample_dataset(tokenized_datasets, args.subsample_factor)
        print(f"Dataset subsampled by factor of {args.subsample_factor}")

    # Print model precision information
    print(f"Model precision: {trainer.original_model.dtype}")
    print(f"\nTraining using {args.training_method} method...")

    # Train PEFT Adapter or full model
    trainer.train(tokenized_datasets)
    print("Training complete!")

    # Save the model
    if args.training_method == "peft":
        trainer.model.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)  # Fixed from save_model to save_pretrained
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model using PEFT/LoRA or full fine-tuning")
    
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name or path to checkpoint")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    parser.add_argument("--dataset-name", type=str, default="knkarthick/dialogsum", help="Dataset name")
    parser.add_argument("--training-method", type=str, default="peft", choices=["peft", "full"], 
                        help="Training method: 'peft' for parameter-efficient fine-tuning, 'full' for full fine-tuning")
    parser.add_argument("--prompt-template", type=str, default="prompts/summarize.txt", help="Path to prompt template")
    parser.add_argument("--output-dir", type=str, default="runs/dialogue-summary/train", help="Output dir")
    parser.add_argument("--subsample", action="store_true", help="Whether to subsample the dataset")
    parser.add_argument("--subsample-factor", type=int, default=100, help="Factor to subsample the dataset by (keep 1/factor)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
