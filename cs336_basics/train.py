import os
import numpy as np
import torch
import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, RoPE
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import (
    cross_entropy,
    clip_gradient
)
from cs336_basics.utils import save_checkpoint, load_checkpoint

class TextDataset:
    def __init__(self, data_path: str, vocab_size: int):
        self.data_path = data_path
        self.vocab_size = vocab_size
        try:
            self.dataset = np.load(data_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred while loading data from {data_path}: {e}")

    def get_batch(self, batch_size:int, context_length: int, device: str = "cpu"):
        return get_batch(self.dataset, batch_size, context_length, device)



# 
# import sys
# import json
# import math
# 
# from datetime import datetime

# 

# 
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import swanlab
    SWABLAB_AVAILABLE = True
except ImportError:
    SWABLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Install with 'pip install swanlab' to use logging feature.")

from tqdm import tqdm


class Logger:
    def __init__(self, project="cs336", run_name="", config=None):
        print(config)
    
    def log(self, data):
        print(data)
    
    def finish(self):
        pass

class SwanlabLogger:
    def __init__(self, project="cs336", run_name="", config=None):
        swanlab.init(
            project=project,
            experiment_name=run_name,
            config = config
        )
        self.step = 0
    def log(self, data):
        swanlab.log(data, step=self.step)
        self.step += 1
    
    def finish(self):
        pass

# ==================== Hyperparameters Configuration ==================== #

class Config:
    """Hyperparameters configuration class"""
    # Model parameters
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    num_heads: int = 16
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: float = 10000.0
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data parameters
    data_path: str = "./data/train.txt"
    val_data_path: Optional[str] = "./data/val.txt"
    val_interval: int = 1000
    
    # Checkpoint parameters
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 5000
    save_best: bool = True
    
    # Logger
    log_interval: int = 100
    use_wandb: bool = False
    wandb_project: str = "cs336"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create configuration from command line arguments"""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        # Filter out private attributes and methods
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def __repr__(self):
        lines = ["=" * 50, "Configuration:", "=" * 50]
        for key, value in self.__dict__.items():
            lines.append(f"  {key:25} = {value}")
        lines.append("=" * 50)
        return "\n".join(lines)

# ==================== Trainer ====================

class Trainer:
    """Trainer supporting checkpoint saving, logging, etc."""
    
    def __init__(self, config: Config, model, optimizer,
                 scheduler, train_loader, val_loader = None, logger = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    
    def save_checkpoint(self, is_best: bool = False):
        
        # Regular save
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.current_step}.pt"
        save_checkpoint(self.model, self.optimizer, self.current_step, checkpoint_path)
        
        # Save best model
        if is_best and self.config.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            save_checkpoint(self.model, self.optimizer, self.current_step, best_path)
            print(f"Saved best model to {best_path}")
        
        # Delete old checkpoints (keep only the latest 5)
        old_checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(old_checkpoints) > 5:
            for old in old_checkpoints[:-5]:
                old.unlink()
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """load checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return
        
        self.current_step = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        
        print(f"Loaded checkpoint from step {self.current_step}")
    
    def log_metrics(self, loss: float, lr: float, mode: str = "train"):
        """Log metrics to console and wandb"""

        log_data = {
            f"{mode}/loss": loss,
            f"{mode}/lr": lr,
            "train/step": self.current_step,
        }
        if mode == "train":
            log_data["train/epoch"] = self.current_epoch
        self.logger.log(log_data)
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        input_ids = batch[0].to(self.config.device)
        labels = batch[1].to(self.config.device)
        
        predicted = self.model(input_ids)
        loss = cross_entropy(predicted, labels)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        # Gradient accumulation
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            clip_gradient(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self) -> float:
        """Validation loop"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for step in tqdm(range(10), desc="Validating"):
                batch = self.val_loader.get_batch(
                    self.config.batch_size, 
                    self.config.context_length, 
                    self.config.device
                )
            
                input_ids = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)
                
                predicted = self.model(input_ids)
                loss = cross_entropy(predicted, labels)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.model.train()
        
        print(f"Validation | Step {self.current_step}: Loss = {avg_loss:.4f}")
        
        if self.logger:
            self.logger.log({"val/loss": avg_loss, "train/step": self.current_step})
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print(self.config)
        print(f"Starting training on {self.config.device}")
        print(f"Checkpoints will be saved to {self.checkpoint_dir}")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        pbar = tqdm(total=self.config.total_steps, desc="Training", unit="step")
    
        for step in range(self.config.total_steps):
            batch = self.train_loader.get_batch(
                self.config.batch_size, 
                self.config.context_length, 
                self.config.device
            )
            loss = self.train_step(batch)
            self.current_step += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{current_lr}'
            })
            

            self.log_metrics(loss, current_lr, mode="train")
            
            # validate
            if self.val_loader and self.current_step % self.config.val_interval == 0:
                val_loss = self.validate()
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                
                if self.current_step % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            elif self.current_step % self.config.save_interval == 0:
                self.save_checkpoint()
            
            
            if self.current_step >= self.config.total_steps:
                break
        
        # last save
        self.save_checkpoint(is_best=True)
        print(f"Training completed! Best validation loss: {self.best_loss:.4f}")
        
        if self.logger:
            self.logger.finish()


# ==================== Main Function ==================== #

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward network dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta base frequency")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--total_steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="./data/train.txt", help="Training data path")
    parser.add_argument("--val_data_path", type=str, default="./data/val.txt", help="Validation data path")

    # Checkpoint parameters
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    
    
    # Logging parameters
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint interval")
    parser.add_argument("--save_best", action="store_true", default=True, help="Save best model checkpoint based on validation metric")
    parser.add_argument("--val_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="cs336", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional, auto-generated if not provided)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config.from_args(args)
    
    # Create dataset and dataloader
    print("Loading datasets...")
    train_dataset = TextDataset(config.data_path, config.vocab_size)
    val_dataset = None
    if config.val_data_path and os.path.exists(config.val_data_path):
        val_dataset = TextDataset(config.val_data_path, config.vocab_size)
    

    # Create model, optimizer, scheduler
    print("Building model...")
    rope = RoPE(config.rope_theta, config.d_model // config.num_heads, config.context_length)
    model = TransformerLM(
        config.vocab_size,
        config.context_length, 
        config.num_layers, 
        config.d_model, 
        config.num_heads, 
        config.d_ff, 
        rope
    )
    model.to(config.device)

    # Use AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = None
    
    if config.use_wandb:
        logger = SwanlabLogger(config.wandb_project, config.wandb_run_name, config.to_dict())
    else:
        logger = Logger(config.wandb_project, config.wandb_run_name, config.to_dict())

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_dataset,
        val_loader=val_dataset,
        logger=logger
    )
    
    # Resume training
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()