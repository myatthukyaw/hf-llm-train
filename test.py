import argparse
import os

from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.utils import load_prompt


def print_comparison(base_output: str, finetuned_output: str, ground_truth: str, model_type: str) -> None:
    """Print a comparison of base model, fine-tuned model, and ground truth."""
    print("\n" + "="*100)
    print(f"COMPARISON (Model type: {model_type})")
    print("="*100)
    
    print("\nBASE MODEL OUTPUT:")
    print("-"*80)
    print(base_output)
    
    print("\nFINE-TUNED MODEL OUTPUT:")
    print("-"*80)
    print(finetuned_output)
    
    print("\nGROUND TRUTH:")
    print("-"*80)
    print(ground_truth)
    
    print("\n" + "="*100)

def is_peft_model(model_path: str) -> bool:
    """Detect if a model is a PEFT model by checking for adapter_config.json."""
    return os.path.exists(os.path.join(model_path, "adapter_config.json"))

def is_checkpoint_path(model_path: str) -> bool:
    """Check if the model path looks like a checkpoint directory."""
    return "checkpoint" in model_path or "run-" in model_path

def load_tokenizer(model_path: str, base_model: str = "google/flan-t5-base") -> AutoTokenizer:
    """Load tokenizer with fallback to base model if not found."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError) as e:
        print(f"Could not load tokenizer from {model_path}. Error: {str(e)}")
        print(f"Falling back to base model tokenizer: {base_model}")
        return AutoTokenizer.from_pretrained(base_model)

def generate_summary(model, tokenizer, prompt: str) -> str:
    """Generate a summary using the given model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = tokenizer.decode(
        model.generate(input_ids=inputs["input_ids"], max_length=100)[0],
        skip_special_tokens=True
    )
    return outputs

def load_finetuned_model(args, base_tokenizer):
    """ Load the fine-tuned model """
    if args.model_type == "peft":
        # Create a separate base model instance for PEFT
        # if don't use the separate model ...
        peft_base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        finetuned_model = PeftModel.from_pretrained(peft_base_model, args.model)
        
        # Use the same tokenizer as the base model
        finetuned_tokenizer = base_tokenizer
    else:
        # Regular model loading
        print(f"Loading fine-tuned model: {args.model}")
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

        # Try to load tokenizer with fallback
        finetuned_tokenizer = load_tokenizer(args.model, args.base_model)
    
    return finetuned_model, finetuned_tokenizer

def main(args: argparse.Namespace) -> None:
    
    dataset = load_dataset(args.dataset_name)
    
    # Auto-detect model type if not specified
    if args.model_type is None:
        if is_peft_model(args.model):
            args.model_type = "peft"
            print(f"Detected PEFT/LoRA model: {args.model}")
        else:
            args.model_type = "regular"
            print(f"Using regular model: {args.model}")
    
    # Load the base model and tokenizer for comparison
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load finetuned model
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        args, base_tokenizer
    )
    finetuned_model.eval()

    # Select a test example
    dialog = dataset["test"][args.index]["dialogue"]
    ground_truth = dataset["test"][args.index]["summary"]

    prompt = load_prompt(args.prompt_template, dialogue=dialog)
    print("Generating summaries...")
    base_output = generate_summary(base_model, base_tokenizer, prompt)
    finetuned_output = generate_summary(finetuned_model, finetuned_tokenizer, prompt)
    print_comparison(base_output, finetuned_output, ground_truth, args.model_type)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Path to finetuned checkpoint")
    parser.add_argument("--base-model", type=str, default="google/flan-t5-base", 
                        help="Base model for PEFT or tokenizer fallback (default: google/flan-t5-base)")
    parser.add_argument("--dataset-name", type=str, default="knkarthick/dialogsum", help="Dataset name")
    parser.add_argument("--prompt-template", type=str, default="prompts/summarize.txt", help="Path to prompt template")
    parser.add_argument("--index", type=int, default=200, help="Index of the test example")
    parser.add_argument("--model-type", type=str, choices=["regular", "peft"], default=None, 
                        help="Type of model (regular or peft). If not specified, will be auto-detected.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 
    main(args)
