# Finetuning LLMs
This repository contains scripts for efficiently fine-tuning Large Language Models (LLMs).

## Feats

- **Training**: Fine-tune pre-trained language models with options for full fine-tuning or PEFT/LoRA
- **Evaluation**: Evaluate model performance with multiple metrics including ROUGE and BLEU
- **Testing**: Compare base and fine-tuned models on test examples
- **Precision Options**: Support for different precision formats (float16, bfloat16, float32, float64)

## Requirements

```
pip install -r requirements.txt
```

## Usage

### Finetuning

```bash
# Full-finetuning
python train.py --model google/flan-t5-base --dataset-name knkarthick/dialogsum --training-method full
# Or

# Parameter-Efficient Fine-Tuning - LORA
python train.py --model google/flan-t5-base --dataset-name knkarthick/dialogsum --training-method peft
```

Options:
- `--training-method`: Choose between `peft` (parameter-efficient) or `full` (full fine-tuning)
- `--precision`: Set precision format (`float16`, `bfloat16`, `float32`, `float64`)
- `--subsample`: Use a subset of the dataset for faster iteration

### RLHF

Instruct fine-tuned LLM -> RLHF -> Human Aligned LLM

RLHF making sure the model outputs the helpfullness and usefullness of input prompt, minimize harmfullness, avoid dangerous topic. 

detoxifying llm

### Evaluation

```bash
python eval.py --base_model google/flan-t5-base --finetuned_model ./dialogue-summary/train/peft-run-
```

Options:
- `--metrics`: Metrics to compute (`rouge`, `bleu`)
- `--num_samples`: Number of samples to evaluate
- `--save_summaries`: Save generated summaries

### Testing

```bash
python test.py --model ./dialogue-summary/train/peft-run- --base-model google/flan-t5-base
```

Options:
- `--index`: Index of the test example
- `--model-type`: Type of model (`regular` or `peft`)


## Default Configuration

The default configuration uses:
- Dataset: `knkarthick/dialogsum` (dialogue summarization)
- Base model: `google/flan-t5-base`
- PEFT/LoRA parameters: r=16, alpha=32, dropout=0.05
