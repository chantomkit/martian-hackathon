"""
Training script for the Safety Judge
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import argparse
from typing import Dict, List

from models.safety_judge import SafetyJudge, SafetyJudgeConfig


class SafetyJudgeTrainer:
    """Trainer for the Safety Judge model"""
    
    def __init__(self, 
                 model: SafetyJudge,
                 tokenizer: AutoTokenizer,
                 train_dataset,
                 val_dataset,
                 config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Only optimize the probe head parameters
        self.optimizer = torch.optim.AdamW(
            self.model.probe_head.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        
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
            labels = torch.tensor([item['label'] for item in batch])
            
            # Tokenize prompts
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors='pt'
            )
            
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
            num_workers=2
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.probe_head.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
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
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of unsafe
                all_labels.extend(labels.cpu().numpy())
        
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
            
            # Early stopping
            if patience_counter >= self.config.get('patience', 5):
                print(f"Early stopping after {epoch + 1} epochs")
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
    parser = argparse.ArgumentParser(description="Train Safety Judge")
    parser.add_argument('--data_dir', type=str, default='./data/prepared/safety_dataset',
                       help='Path to prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--freeze_layers', type=int, default=24,
                       help='Number of layers to freeze in base model')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'weight_decay': 0.01,
        'patience': 3,
        'output_dir': args.output_dir,
        'use_wandb': args.use_wandb
    }
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    
    # Initialize model
    model_config = SafetyJudgeConfig(
        freeze_layers=args.freeze_layers,
        max_length=args.max_length
    )
    
    print("Initializing Safety Judge model...")
    model = SafetyJudge(model_config)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create trainer
    trainer = SafetyJudgeTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        val_dataset=dataset['validation'],
        config=config
    )
    
    # Train
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main() 