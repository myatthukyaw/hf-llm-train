from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from peft import TaskType

@dataclass
class TrainingConfig:
    """Configuration for standard training."""
    output_dir : str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 5e-5
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    seed: int = 42
    
    def to_dict(self) -> dict:
        """Convert config to dictionary.
        This is to pass the arguments into transformers.TrainingArguments"""
        return {
            "output_dir" : self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            # "max_length": self.max_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "fp16": self.fp16,
            "seed": self.seed
        }


@dataclass
class LoraConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 32
    bias: str = "none"
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None # field(default_factory=lambda: ["q", "v"])
    task_type: TaskType  = TaskType.SEQ_2_SEQ_LM # FLAN-T5
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "r": self.r,
            "bias": self.bias,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "task_type": self.task_type,
        }


@dataclass
class PpoConfig:
    """Configuration for PPO training."""
    output_dir: str
    learning_rate: float = 1.4e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "mini_batch_size": self.mini_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
