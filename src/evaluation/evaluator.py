import os
from typing import List, Tuple
import pandas as pd
import torch
import numpy as np
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from src.evaluation.metrics import EvalMetrics
from src.utils.utils import format_prompt, load_prompt_template


class Evaluator(EvalMetrics):
    """Dialogue Summarization Dataset Evaluator"""  
    def __init__(
        self,
        prompt_template: str, 
        device: str, 
        max_new_tokens: int =200, 
        batch_size: int = 4) -> None:
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.template = load_prompt_template(prompt_template)

    def _load_tokenizer(self, model_path: str, model_name: str = None) -> AutoTokenizer:
        """
        Load tokenizer with fallback to parent directory or base model.
        Args:
            model_path: Path to the model directory
            model_name: Optional base model name to fall back to
        Returns:
            AutoTokenizer
        """
        if model_name:
            return AutoTokenizer.from_pretrained(model_name)
        else:
            try:
                return AutoTokenizer.from_pretrained(model_path)
            except OSError:
                if "checkpoint-" in model_path:
                    # If model_path is a checkpoint path, try the parent directory.
                    parent_dir = os.path.dirname(os.path.dirname(model_path))
                    print(f"Tokenizer not found in {model_path}. Trying parent dir: {parent_dir}")
                    return AutoTokenizer.from_pretrained(parent_dir)
                elif model_name:
                    print(f"Falling back to base model tokenizer: {model_name}")
                    return AutoTokenizer.from_pretrained(model_name)
            raise

    def load_model_and_tokenizer(self, model_path: str, base_model: str = None) -> tuple:
        """
        Load the model and tokenizer.
        Args:
            model_path: Path to the model directory
            base_model: Base model name for PEFT adapters (optional)
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Check if it's a PEFT adapter
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                print(f"Loading PEFT model from {model_path}")
                return self._load_peft_model(model_path, base_model)
            else:
                print(f"Loading full model from {model_path}")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
                tokenizer = self._load_tokenizer(model_path)
                return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")
    
    def _load_peft_model(self, model_path: str, base_model: str = None) -> tuple:
        """
        Load a PEFT model and its tokenizer.
        Args:
            model_path: Path to the PEFT model directory
            base_model: Base model name (will be read from config if None)
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        if base_model is None:
            # Get base model from PEFT config
            config = PeftConfig.from_pretrained(model_path)
            base_model = config.base_model_name_or_path
            print(f"Using base model from PEFT config: {base_model}")
        
        # First try to load the tokenizer from the adapter or parent directory,
        # then fall back to the base model if needed
        tokenizer = self._load_tokenizer(model_path, base_model)
        
        # Load base model and adapter
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(self.device)
        model = PeftModel.from_pretrained(base, model_path).to(self.device)
        
        return model, tokenizer
    
    def generate_summaries(self, 
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        dialogues: List[str]) -> List[str]:
        """
        Generate summaries for a list of dialogues.
        Args:
            dialogues: List of dialogue texts to summarize        
        Returns:
            List of generated summaries
        """
        summaries = []

        for i in range(0, len(dialogues), self.batch_size):
            batch_dialogues = dialogues[i:i+self.batch_size]
            # Format prompts with template
            prompts = [format_prompt(self.template, dialogue=d) for d in batch_dialogues]
            summaries.extend(
                self._batch_inference(model, tokenizer, prompts)
            )
            
        return summaries
    
    def _batch_inference(self, 
        model: AutoModelForSeq2SeqLM, 
        tokenizer : AutoTokenizer, 
        prompts: list) -> List[str]:
        """
        Process a batch of dialogues for summarization.
        """
        
        # Tokenize inputs
        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        # Generate summaries
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=GenerationConfig(max_new_tokens=self.max_new_tokens)
            )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def evaluate_summaries(self, 
        preds: list, refs: list, 
        metrics: list= ["rouge","bleu"]) -> dict:
        """Evaluate summaries using multiple metrics."""
        results = {}
        
        if "rouge" in metrics:
            rouge_results = self.compute_rouge(preds, refs)
            results.update(rouge_results)
        if "bleu" in metrics:
            bleu_results = self.compute_bleu(preds, refs)
            results.update(bleu_results)
        
        return results

    def calculate_improvements(self, baseline_results: dict, new_results: dict)-> tuple:
        """Calculate absolute and percentage improvements."""
        abs_improvement = {}
        pct_improvement = {}
        
        for key in baseline_results:
            if key in new_results:
                # Handle both scalar and list-type metric values
                if isinstance(baseline_results[key], list) and isinstance(new_results[key], list):
                    baseline_array = np.array(baseline_results[key])
                    new_array = np.array(new_results[key])
                    
                    # Calculate improvements as numpy arrays
                    if baseline_array.size == new_array.size:
                        abs_diff = new_array - baseline_array
                        abs_improvement[key] = float(np.mean(abs_diff))
                        
                        # Calculate percentage improvement
                        nonzero_mask = baseline_array != 0 # avoiding division by zero
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

    def save_summaries(self, 
        dialogues: List[str], 
        human_summaries: List[str], 
        base_summaries: List[str], 
        finetuned_summaries: List[str], 
        save_dir: str) -> None:
        """Save the generated summaries to a CSV file. """
        df = pd.DataFrame({
            "dialogue": dialogues,
            "human_summary": human_summaries,
            "base_model_summary": base_summaries,
            "finetuned_model_summary": finetuned_summaries
        })
        summaries_path = os.path.join(save_dir, "summaries.csv")
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(summaries_path, index=False)
        print(f"Saved summaries to {summaries_path}")