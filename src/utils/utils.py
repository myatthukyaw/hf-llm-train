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


def print_trainable_model_params(model) -> str:
    """
    Function to analyze model parameters to understand how many parameters are trainable vs. frozen
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()  # Count all parameters
        if param.requires_grad:
            trainable_model_params += param.numel()  # Count only trainable parameters

    percentage_trainable = (trainable_model_params / all_model_params) * 100
    return (
        f"Trainable model parameters: {trainable_model_params}, \n"
        f"All model parameters: {all_model_params}, \n"
        f"Percentage of trainable model parameters: {percentage_trainable:.2f}%"
    )

