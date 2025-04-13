import argparse
import evaluate
import numpy as np
import pandas as pd
import torch
import os
import json
import time
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
from typing import Dict, List, Optional, Tuple, Union

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer( model_path: str, device, base_model: str = None ) -> tuple:
    """Load model and tokenizer with proper error handling and device placement."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Loading PEFT model from {model_path}")
            if base_model is None:
                # Try to get base model from PEFT config
                config = PeftConfig.from_pretrained(model_path)
                base_model = config.base_model_name_or_path
                print(f"Using base model from PEFT config: {base_model}")
            
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
            model = PeftModel.from_pretrained(base, model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            print(f"Loading full model from {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")

def create_prompt(dialogue: str, prompt_template: str) -> str:
    """Create prompt from dialogue based on template."""
    return prompt_template.format(dialogue=dialogue)

def generate_summaries_batch(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    dialogues: List[str],
    prompt_template: str,
    batch_size: int = 4,
    max_new_tokens: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list:
    """Generate summaries in batches for efficiency."""
    summaries = []
    
    for i in range(0, len(dialogues), batch_size):
        batch_dialogues = dialogues[i:i+batch_size]
        prompts = [create_prompt(d, prompt_template) for d in batch_dialogues]
        
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=GenerationConfig(max_new_tokens=max_new_tokens)
            )
            
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(batch_summaries)
    return summaries

def evaluate_summaries(preds: list, refs: list, metrics: list = ["rouge","bleu"]) -> dict:
    """Evaluate summaries using multiple metrics."""
    results = {}
    
    if "rouge" in metrics:
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(
            predictions=preds, 
            references=refs,
            use_stemmer=True,
            use_aggregator=True
        )
        results.update(rouge_results)
    
    if "bleu" in metrics:
        bleu = evaluate.load("bleu")
        # BLEU expects a list of references for each prediction
        references_for_bleu = [[ref] for ref in refs]
        bleu_results = bleu.compute(predictions=preds, references=references_for_bleu)
        results.update(bleu_results)
    
    return results


def calculate_improvements(baseline_results: dict, new_results: dict)-> tuple:
    """Calculate absolute and percentage improvements."""
    abs_improvement = {}
    pct_improvement = {}
    
    for key in baseline_results:
        if key in new_results:
            # Handle both scalar and list-type metric values
            if isinstance(baseline_results[key], list) and isinstance(new_results[key], list):
                # Convert lists to numpy arrays for element-wise operations
                baseline_array = np.array(baseline_results[key])
                new_array = np.array(new_results[key])
                
                # Calculate improvements as numpy arrays
                if baseline_array.size == new_array.size:
                    abs_diff = new_array - baseline_array
                    
                    # Store the mean improvement as a scalar
                    abs_improvement[key] = float(np.mean(abs_diff))
                    
                    # Calculate percentage improvement, avoiding division by zero
                    nonzero_mask = baseline_array != 0
                    if any(nonzero_mask):
                        pct_diff = np.zeros_like(baseline_array, dtype=float)
                        pct_diff[nonzero_mask] = (abs_diff[nonzero_mask] / baseline_array[nonzero_mask]) * 100
                        pct_improvement[key] = float(np.mean(pct_diff[nonzero_mask]))
                    else:
                        pct_improvement[key] = float('inf') if np.sum(abs_diff) > 0 else 0.0
            else:
                # Handle scalar values
                abs_improvement[key] = new_results[key] - baseline_results[key]
                if baseline_results[key] != 0:
                    pct_improvement[key] = (abs_improvement[key] / baseline_results[key]) * 100
                else:
                    pct_improvement[key] = float('inf') if abs_improvement[key] > 0 else 0
    
    return abs_improvement, pct_improvement


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

def parse_arugments()-> ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dialogue summarization models")
    
    # Dataset params
    parser.add_argument("--dataset", type=str, default="knkarthick/dialogsum", help="Dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    
    # Model params
    parser.add_argument("--base_model", type=str, default="google/flan-t5-base", help="Base model to evaluate")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Fine-tuned model to evaluate")
    parser.add_argument("--prompt_template", type=str, 
                        default="Summarize the following conversation.\n\n{dialogue}\n\nSummary:", 
                        help="Prompt template for generation")
    
    # Generation params
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    
    # Evaluation params
    parser.add_argument("--metrics", nargs="+", default=["rouge", "bleu"], help="Metrics to compute")
    parser.add_argument("--save_dir", type=str, default="runs/dialogue-summary/eval", help="Directory to save results")
    parser.add_argument("--save_summaries", action="store_true", help="Whether to save generated summaries")
    
    # Other params
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    
    return parser.parse_args()

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
    
    # Load base model and tokenizer
    print(f"Loading base model {args.base_model}...")
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model, device=args.device)
    
    # Load finetuned model and tokenizer
    print(f"Loading fine-tuned model {args.finetuned_model}...")
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(
        args.finetuned_model, args.device, args.base_model
    )
    
    start_time = time.time()
    
    # Generate summaries with base model
    print("Generating summaries with base model...")
    base_summaries = generate_summaries_batch(
        base_model, 
        base_tokenizer, 
        dialogues,
        args.prompt_template,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # Generate summaries with finetuned model
    print("Generating summaries with finetuned model...")
    finetuned_summaries = generate_summaries_batch(
        finetuned_model, 
        finetuned_tokenizer, 
        dialogues,
        args.prompt_template,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Create DataFrame with summaries
    df = pd.DataFrame({
        "dialogue": dialogues,
        "human_summary": human_summaries,
        "base_model_summary": base_summaries,
        "finetuned_model_summary": finetuned_summaries
    })
    
    # Save summaries
    if args.save_summaries:
        summaries_path = os.path.join(args.save_dir, "summaries.csv")
        os.makedirs(args.save_dir, exist_ok=True)
        df.to_csv(summaries_path, index=False)
        print(f"Saved summaries to {summaries_path}")
    
    # Evaluate base model
    print("Evaluating base model...")
    base_results = evaluate_summaries(base_summaries, human_summaries, args.metrics)
    
    # Evaluate finetuned model
    print("Evaluating finetuned model...")
    finetuned_results = evaluate_summaries(finetuned_summaries, human_summaries, args.metrics)
    
    # Calculate improvements
    abs_improvement, pct_improvement = calculate_improvements(base_results, finetuned_results)
    
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


if __name__ == "__main__":
    main()