import argparse
import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from datasets import load_dataset
from src.evaluation.evaluator import Evaluator
from src.utils.utils import format_prompt, load_prompt_template


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results(results: Dict, save_dir: str, filename: str = None) -> str:
    """Save evaluation results to a file."""
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{timestamp}.json"
    
    file_path = os.path.join(save_dir, filename)
    
    # Convert any non-serializable values to strings
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray) or isinstance(value, np.float64):
            serializable_results[key] = float(value)
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            serializable_results[key] = [float(v) for v in value]
        else:
            serializable_results[key] = value
    
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return file_path

def main() -> None:

    args = parse_arugments()
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset)
    
    # Select samples
    eval_samples = dataset[args.split]
    if args.num_samples < len(eval_samples):
        eval_samples = eval_samples.select(range(args.num_samples))
    
    dialogues = eval_samples["dialogue"]
    human_summaries = eval_samples["summary"]

    evaluator = Evaluator(
        device=args.device, 
        max_new_tokens=args.max_new_tokens,
        prompt_template=args.prompt_template,
    )
    
    # Load base model and tokenizer
    print(f"Loading base model {args.base_model}...")
    base_model, base_tokenizer = evaluator.load_model_and_tokenizer(
        args.base_model
    )
    
    # Load finetuned model and tokenizer
    print(f"Loading fine-tuned model {args.finetuned_model}...")
    finetuned_model, finetuned_tokenizer = evaluator.load_model_and_tokenizer(
        args.finetuned_model, base_model = args.base_model
    )
    
    start_time = time.time()
    
    # Generate summaries with base model
    print("Generating summaries with base model...")
    base_summaries = evaluator.generate_summaries(
        base_model,
        base_tokenizer,
        dialogues
    )
    
    # Generate summaries with finetuned model
    print("Generating summaries with finetuned model...")
    finetuned_summaries = evaluator.generate_summaries(
        finetuned_model, 
        finetuned_tokenizer, 
        dialogues
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Save summaries
    if args.save_summaries:
        evaluator.save_summaries(
            dialogues, human_summaries, base_summaries, finetuned_summaries, args.save_dir
        )
    
    # Evaluate base model
    print("Evaluating base model...")
    base_results = evaluator.evaluate_summaries(
        base_summaries, human_summaries, args.metrics
    )
    
    # Evaluate finetuned model
    print("Evaluating finetuned model...")
    finetuned_results = evaluator.evaluate_summaries(
        finetuned_summaries, human_summaries, args.metrics
    )
    
    # Calculate improvements
    abs_improvement, pct_improvement = evaluator.calculate_improvements(
        base_results, finetuned_results
    )
    
    # Prepare results for saving
    all_results = {
        "args": vars(args),
        "base_model_results": base_results,
        "finetuned_model_results": finetuned_results,
        "absolute_improvement": abs_improvement,
        "percentage_improvement": pct_improvement,
        "generation_time_seconds": generation_time,
        "num_samples": len(dialogues),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    results_path = save_results(all_results, args.save_dir)
    print(f"Saved results to {results_path}")
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nBase model results:")
    for metric, value in base_results.items():
        if isinstance(value, list):
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nFine-tuned model results:")
    for metric, value in finetuned_results.items():
        if isinstance(value, list):
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nAbsolute improvement:")
    for metric, value in abs_improvement.items():
        if isinstance(value, list):
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nPercentage improvement:")
    for metric, value in pct_improvement.items():
        if isinstance(value, list):
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.2f}%")
    
    print("\n" + "="*50)

def parse_arugments()-> ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dialogue summarization models")
    
    # Dataset params
    parser.add_argument("--dataset", type=str, default="knkarthick/dialogsum", help="Dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    
    # Model params
    parser.add_argument("--base-model", type=str, default="google/flan-t5-base", help="Base model to evaluate")
    parser.add_argument("--finetuned-model", type=str, required=True, help="Fine-tuned model to evaluate")
    parser.add_argument("--prompt-template", type=str, default="prompts/summarize.txt", help="Path to prompt template")
    
    # Generation params
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum number of tokens to generate")
    
    # Evaluation params
    parser.add_argument("--metrics", nargs="+", default=["rouge", "bleu"], help="Metrics to compute")
    parser.add_argument("--save-dir", type=str, default="runs/dialogue-summary/eval", help="Directory to save results")
    parser.add_argument("--save-summaries", action="store_true", help="Whether to save generated summaries")
    
    # Other params
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()