import argparse
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from utils.utils import print_trainable_model_parameters, tokenize_function


def load_model(args):
    """ Load model with appropriate precision """  
    if args.precision == "float16":
        original_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16)
    elif args.precision == "bfloat16":
        original_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    elif args.precision == "float64":
        original_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float64)
    else:  # Default: float32
        original_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    # Choose between PEFT/LoRA and full fine-tuning
    if args.training_method == "peft":
        print("Using PEFT/LoRA for efficient fine-tuning")
        # Setup LoRA/PEFT model for fine-tuning
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(original_model, lora_config)
    else:
        print("Using full fine-tuning")
        model = original_model  # Use the original model for full fine-tuning

    return model, original_model

def main(args):
    # Load dataset and model
    dataset = load_dataset(args.dataset_name)
    model, original_model = load_model(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply tokenization to the dataset
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary']
    )
    print("Tokenization complete!")

    # Set the format to PyTorch tensors
    tokenized_datasets = tokenized_datasets.with_format("torch")
    
    # Subsample if requested
    if args.subsample:
        tokenized_datasets = tokenized_datasets.filter(
            lambda example, index: index % args.subsample_factor == 0, 
            with_indices=True
        )
        print(f"Dataset subsampled by factor of {args.subsample_factor}")

    # Print parameter information
    print(print_trainable_model_parameters(original_model))
    print(print_trainable_model_parameters(model))

    # Train PEFT Adapter or full model
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"{args.training_method}-run-{timestamp}")

    # Configure training arguments based on the model's dtype
    training_args = {
        "output_dir": output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "logging_steps": args.logging_steps,
        "per_device_eval_batch_size": args.eval_batch_size,
        "per_device_train_batch_size": args.train_batch_size,
    }

    # Add precision-specific settings
    if original_model.dtype == torch.float16:
        training_args.update({
            "fp16": True,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        })
    elif original_model.dtype == torch.bfloat16:
        training_args.update({
            "bf16": True,  # Use bfloat16 mixed precision
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        })
    else:
        training_args.update({
            "fp16": False,
            "gradient_accumulation_steps": 1,
        })

    # For full fine-tuning, we might want a different learning rate or weight decay
    if args.training_method == "full":
        training_args.update({
            "learning_rate": args.full_learning_rate,
            "weight_decay": args.weight_decay,
        })

    training_args = TrainingArguments(**training_args)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Print model precision information
    print(f"Model precision: {original_model.dtype}")
    print(f"\nTraining using {args.training_method} method...")
    trainer.train()
    print("Training complete!")

    # Save the model
    if args.save_model:
        model_path = os.path.join(args.save_model_dir, f"{args.training_method}-model")
        if args.training_method == "peft":
            trainer.model.save_pretrained(model_path)
        else:
            trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model using PEFT/LoRA or full fine-tuning")
    
    # Model and dataset arguments
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name or path to checkpoint")
    parser.add_argument("--dataset-name", type=str, default="knkarthick/dialogsum", help="Dataset name")
    
    # Training method
    parser.add_argument("--training-method", type=str, default="peft", choices=["peft", "full"], 
                        help="Training method: 'peft' for parameter-efficient fine-tuning, 'full' for full fine-tuning")
    
    # Precision arguments
    parser.add_argument("--precision", type=str, default="float16", choices=["float32", "float16", "bfloat16", "float64"])
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for PEFT")
    parser.add_argument("--full-learning-rate", type=float, default=5e-5, help="Learning rate for full fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for full fine-tuning")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    
    # LoRA arguments (only used for PEFT)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA r dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA bias type")
    parser.add_argument("--target-modules", type=str, default="q,v,k,o", help="Comma-separated list of target modules for LoRA")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="runs/dialogue-summary/train", help="Output dir")
    parser.add_argument("--save-model", action="store_true", help="Whether to save the model")
    parser.add_argument("--save-model-dir", type=str, default="dialogue-summary-model", help="Directory to save the model")
    
    # Subsampling arguments
    parser.add_argument("--subsample", action="store_true", help="Whether to subsample the dataset")
    parser.add_argument("--subsample-factor", type=int, default=100, help="Factor to subsample the dataset by (keep 1/factor)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
