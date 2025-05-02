from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from src.utils.utils import print_trainable_model_params


class BaseTrainer:
    def __init__(
        self,
        model_name : str,
        train_config : Optional[Dict[str, Any]],
        lora_config: Optional[Dict[str, Any]] = None,
        precision: str = "float32"
    ) -> None:
        self.model_name = model_name
        self.train_config = train_config
        self.lora_config = lora_config
        self.precision = precision
        
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        
    def load_model(self):
        """Initialize the model with appropriate precision."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64
        }
        
        self.original_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype_map.get(self.precision, torch.float16)
        )
        
        if self.lora_config:
            print("Using PEFT/LoRA for efficient fine-tuning")
            lora_config = LoraConfig(**self.lora_config.to_dict())
            self.model = get_peft_model(self.original_model, lora_config)
        else:
            print("Using full fine-tuning")
            self.model = self.original_model
            
    def load_tokenizer(self):
        """Initialize the tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.model_name
        )

    def _add_precision_specific_setting(self, training_args):
        """ Add precision-specific settings. """
        if self.model.dtype == torch.float16:
            training_args.update({
                "fp16": True,
                "fp16_full_eval": True,
                "gradient_accumulation_steps": self.train_config.gradient_accumulation_steps,
            })
        elif self.model.dtype == torch.bfloat16:
            training_args.update({
                "bf16": True, 
                "gradient_accumulation_steps": self.train_config.gradient_accumulation_steps,
            })
        else:
            training_args.update({
                "fp16": False,
                "gradient_accumulation_steps": 1,
            })
            
        return training_args

    def train(self, dataset) -> None:
        """Train the model."""
        print(print_trainable_model_params(self.model))
        
        # Initialize training arguments
        train_args_dict = self.train_config.to_dict()
        train_args_dict = self._add_precision_specific_setting(train_args_dict)
        
        training_args = TrainingArguments(
            **train_args_dict
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"]
        )
        
        # Train the model
        print(f"\nTraining using {'PEFT' if self.lora_config else 'full'} method...")
        trainer.train()
        print("Training complete!")
        
        return trainer
