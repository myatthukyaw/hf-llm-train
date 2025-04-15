def load_prompt_template(prompt_template_path: str) -> str:
    """
    Load a prompt template from a file.
    """
    with open(prompt_template_path, 'r') as f:
        return f.read()

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given variables.
    """
    return template.format(**kwargs)

# For backward compatibility
def load_prompt(prompt_template: str, **kwargs) -> str:
    """
    Load and format a prompt template in one step.
    """
    template = load_prompt_template(prompt_template)
    return format_prompt(template, **kwargs)


def print_trainable_model_parameters(model) -> str:
    """
    Function to analyze model parameters
    This helps understand how many parameters are trainable vs. frozen
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()  # Count all parameters
        if param.requires_grad:
            trainable_model_params += param.numel()  # Count only trainable parameters
    return f"Trainable model parameters: {trainable_model_params}, All model parameters: {all_model_params}, Percentage of trainable model parameters: {trainable_model_params/all_model_params*100:.2f}%"


def tokenize_function(example, tokenizer):
    # Define the prompt structure
    start_prompt = "Summarize the following conversation:\n"
    end_prompt = "\nSummary :"
    
    # Create prompts for each dialogue in the batch
    prompts = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompts, padding="max_length", truncation=True).input_ids
    example['labels'] = tokenizer(example['summary'], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example