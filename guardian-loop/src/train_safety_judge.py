"""
Training script for the Safety Judge
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import argparse
from typing import Dict, List
import multiprocessing

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from models.safety_judge import SafetyJudge, SafetyJudgeConfig

class SafetyDataset(Dataset):
    """Custom dataset for safety data"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item['prompt'],
            'label': item['label']
        }

def load_json_data(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

class SafetyJudgeTrainer:
    """Trainer for the Safety Judge model"""
    
    def __init__(self, 
                 model: SafetyJudge,
                 tokenizer: AutoTokenizer,
                 train_data,
                 val_data,
                 config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = SafetyDataset(train_data)
        self.val_dataset = SafetyDataset(val_data)
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Move model to device and get its dtype
        self.model.to(self.device)
        self.dtype = next(self.model.parameters()).dtype
        print(f"Model dtype: {self.dtype}")
        
        # Setup mixed precision training
        self.use_amp = self.dtype == torch.float16 and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Only optimize the probe head parameters
        self.optimizer = torch.optim.AdamW(
            self.model.probe_head.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="guardian-loop-safety-judge",
                config=config,
                name=config.get('run_name', 'safety-judge-training')
            )
    
    def _create_dataloader(self, dataset, shuffle=True):
        """Create a DataLoader for the dataset"""
        def collate_fn(batch):
            prompts = [item['prompt'] for item in batch]
            labels = torch.tensor([item['label'] for item in batch], device=self.device, dtype=torch.long)
            
            # Tokenize prompts using encode_plus instead of direct call
            encoded = self.tokenizer.batch_encode_plus(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors='pt'
            )
            
            # Move tensors to GPU with correct dtypes:
            # - input_ids: long (for embedding lookup)
            # - attention_mask: bool
            encoded = {
                'input_ids': encoded['input_ids'].to(device=self.device, dtype=torch.long),
                'attention_mask': encoded['attention_mask'].to(device=self.device, dtype=torch.bool),
            }
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': labels
            }
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Disable multiprocessing for now to avoid CUDA issues
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Print GPU memory usage at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache at start of epoch
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(batch['input_ids'], batch['attention_mask'])
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, batch['labels'])
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.probe_head.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress bar with GPU memory info
            if torch.cuda.is_available():
                mem_info = f"GPU mem: {torch.cuda.memory_allocated() / 1024**2:.0f}MB"
                progress_bar.set_postfix({'loss': loss.item(), 'mem': mem_info})
            else:
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Explicitly clear some memory every few batches
            if batch_idx % 10 == 0:
                del outputs, logits, loss
                torch.cuda.empty_cache()
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(batch['input_ids'], batch['attention_mask'])
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate loss
                    loss = self.criterion(logits, batch['labels'])
                    total_loss += loss.item()
                    
                    # Get predictions and probabilities
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                
                # Move results to CPU and convert to numpy
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of unsafe
                all_labels.extend(batch['labels'].cpu().numpy())
                
                # Clear memory
                del outputs, logits, probs, preds
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _calculate_metrics(self, labels, preds, probs=None):
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=0  # 0 = unsafe
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC if probabilities available
        if probs is not None:
            try:
                # Convert labels to match probability interpretation
                # probs are P(unsafe), so label 0 (unsafe) = 1 for AUC
                auc_labels = 1 - np.array(labels)
                auc = roc_auc_score(auc_labels, probs)
                metrics['auc'] = auc
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train(self):
        """Main training loop"""
        best_val_metric = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train metrics: {train_metrics}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val metrics: {val_metrics}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics.get('auc', 0),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save best model
            val_metric = val_metrics.get('auc', val_metrics['f1'])
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                self.save_checkpoint(
                    Path(self.config['output_dir']) / 'best_model.pt',
                    epoch,
                    val_metrics
                )
                print(f"Saved best model with validation AUC/F1: {val_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.get('patience', 3):
                    print("Early stopping triggered")
                    break
        
        # Save final model
        self.save_checkpoint(
            Path(self.config['output_dir']) / 'final_model.pt',
            epoch,
            val_metrics
        )
        
        print("\nTraining completed!")
        print(f"Best validation metric: {best_val_metric:.4f}")
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'model_config': self.model.config
        }
        
        torch.save(checkpoint, path)
        
        # Also save metrics as JSON for easy inspection
        metrics_path = path.parent / f"{path.stem}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


def main():
    import os
    from huggingface_hub import login
    import dotenv
    
    dotenv.load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    parser = argparse.ArgumentParser(description='Train the Safety Judge model')
    parser.add_argument('--data_dir', type=str, default='./data/prepared',
                       help='Path to prepared dataset directory')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Name of the pretrained model to use')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Whether to use Weights & Biases for logging')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Enable gradient checkpointing to save memory')
    
    args = parser.parse_args()
    
    # Load the data
    train_data = load_json_data(Path(args.data_dir) / 'train.json')
    val_data = load_json_data(Path(args.data_dir) / 'val.json')
    test_data = load_json_data(Path(args.data_dir) / 'test.json')
    
    # Initialize model and tokenizer
    config = SafetyJudgeConfig(
        base_model=args.base_model,
        max_length=args.max_length
    )
    model = SafetyJudge(config)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Configure padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Enable memory optimizations
    if args.gradient_checkpointing and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        model.base_model.enable_input_require_grads()
    
    # Use memory efficient attention if available
    if hasattr(model.base_model.config, 'use_memory_efficient_attention'):
        model.base_model.config.use_memory_efficient_attention = True
    
    # Configure model to use flash attention if available
    if hasattr(model.base_model.config, 'attn_implementation'):
        model.base_model.config.attn_implementation = "flash_attention_2"
    
    # Create trainer
    trainer_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_length': args.max_length,
        'use_wandb': args.use_wandb,
        'output_dir': args.output_dir
    }
    
    trainer = SafetyJudgeTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config
    )
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main() 