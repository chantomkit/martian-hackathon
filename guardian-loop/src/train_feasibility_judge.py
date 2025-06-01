"""
Training script for the Feasibility Judge using prompting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import argparse
from typing import Dict, List, Union, Any, Optional, cast, TypeVar
import multiprocessing
import plotly.graph_objects as go
from torch.cuda.amp import autocast, GradScaler

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from models.feasibility_judge import FeasibilityJudge, FeasibilityJudgeConfig

TokenizerType = TypeVar('TokenizerType', bound=PreTrainedTokenizer)

class FeasibilityDataset(Dataset):
    """Custom dataset for feasibility data"""
    def __init__(self, data_or_path: Union[str, List[Dict]], tokenizer: TokenizerType, config: FeasibilityJudgeConfig):
        if isinstance(data_or_path, str):
            # Load from JSON file
            with open(data_or_path, 'r') as f:
                data = json.load(f)
                # Ensure data is a list
                self.data = data if isinstance(data, list) else [data]
        else:
            # Use data directly
            self.data = data_or_path
        
        self.tokenizer = tokenizer
        self.config = config
        
        # Get True/False token ids
        true_tokens = tokenizer.encode(" True", add_special_tokens=False)
        false_tokens = tokenizer.encode(" False", add_special_tokens=False)
        self.true_token_id = true_tokens[0]
        self.false_token_id = false_tokens[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Ensure we have the required fields
        if 'prompt' not in item or 'answer' not in item or 'label' not in item:
            raise ValueError(f"Data item at index {idx} missing required fields. Expected 'prompt', 'answer', and 'label', got: {item.keys()}")
        
        # Format prompt with template
        prompt = self.config.prompt_template.format(prompt=item['prompt'], answer=item['answer'])
        
        # Tokenize prompt
        inputs = self.tokenizer.encode(
            prompt,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            return_attention_mask=True,
            add_special_tokens=True
        )
        
        # Create attention mask
        attention_mask = [1] * len(inputs)
        
        # Create labels tensor - set True/False token as target
        target_token = self.true_token_id if item['label'] == 1 else self.false_token_id
        labels = torch.full_like(torch.tensor(inputs), -100)
        labels[-1] = target_token
        
        return {
            'input_ids': torch.tensor(inputs),
            'attention_mask': torch.tensor(attention_mask),
            'labels': labels,
            'raw_label': item['label']
        } 

class FeasibilityJudgeTrainer:
    """Trainer for the Feasibility Judge model"""
    
    def __init__(self, 
                 model: FeasibilityJudge,
                 tokenizer: TokenizerType,
                 train_data: List[Dict],
                 val_data: List[Dict],
                 config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        
        # Create datasets from raw data
        self.train_dataset = FeasibilityDataset(train_data, tokenizer, model.config)
        self.val_dataset = FeasibilityDataset(val_data, tokenizer, model.config)
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Setup mixed precision training
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            print("Using automatic mixed precision training")
            # Ensure model is in mixed precision compatible format
            self.model = self.model.float()  # Start with FP32
        else:
            print("Using standard float32 training")
        
        # Move model to device
        self.model.to(self.device)
        
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Initialize MI visualizer if requested
        self.mi_visualizer = None
        if config.get('visualize_during_training', False):
            from mi_tools.visualization import FeasibilityJudgeMIVisualizer
            self.mi_visualizer = FeasibilityJudgeMIVisualizer(model, tokenizer)
            self.visualization_dir = Path(config['output_dir']) / 'training_visualizations'
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            print(f"📊 MI visualizations will be saved to {self.visualization_dir}")
        
        # Get trainable parameters
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Training: {name}")
        
        print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Optimizer with single learning rate
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        
        # Cosine learning rate scheduler with warmup
        num_training_steps = len(self.train_loader) * config['num_epochs']
        num_warmup_steps = int(0.1 * num_training_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="guardian-loop-feasibility-judge",
                config=config,
                name=config.get('run_name', 'feasibility-judge-training')
            )
    
    def _create_dataloader(self, dataset: FeasibilityDataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for the dataset"""
        def collate_fn(batch):
            # Pad sequences in batch
            max_len = max(len(item['input_ids']) for item in batch)
            
            input_ids = []
            attention_mask = []
            labels = []
            raw_labels = []
            
            for item in batch:
                # Pad input_ids and attention_mask
                pad_len = max_len - len(item['input_ids'])
                input_ids.append(torch.cat([
                    item['input_ids'],
                    torch.zeros(pad_len, dtype=torch.long)
                ]))
                attention_mask.append(torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len)
                ]))
                
                # Pad labels
                labels.append(torch.cat([
                    item['labels'],
                    torch.full((pad_len,), -100, dtype=torch.long)  # Ignore index for loss
                ]))
                
                raw_labels.append(item['raw_label'])
            
            return {
                'input_ids': torch.stack(input_ids).to(self.device),
                'attention_mask': torch.stack(attention_mask).to(self.device),
                'labels': torch.stack(labels).to(self.device),
                'raw_labels': torch.tensor(raw_labels).to(self.device)
            }
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Disable multiprocessing for now
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Gradient accumulation for effective larger batch size
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass with automatic mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps or at the end
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Step optimizer and scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item() * accumulation_steps
            
            # Get predictions from logits
            with torch.no_grad():
                logits = outputs['logits']
                true_false_logits = logits[:, [self.train_dataset.true_token_id, self.train_dataset.false_token_id]]
                preds = (torch.softmax(true_false_logits, dim=-1)[:, 0] > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['raw_labels'].cpu().numpy())
            
            # Update progress bar with GPU memory info
            if torch.cuda.is_available():
                mem_info = f"GPU mem: {torch.cuda.memory_allocated() / 1024**2:.0f}MB"
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps, 'mem': mem_info})
            else:
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
            
            # Clear memory periodically
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
        
        # For logging examples
        num_examples_to_log = 4
        logged_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Get predictions from logits
                logits = outputs['logits']
                true_false_logits = logits[:, [self.val_dataset.true_token_id, self.val_dataset.false_token_id]]
                probs = torch.softmax(true_false_logits, dim=-1)
                preds = (probs[:, 0] > 0.5).long()
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 0].cpu().numpy())  # Probability of True/feasible
                all_labels.extend(batch['raw_labels'].cpu().numpy())
                
                # Log a few examples from the first batch
                if batch_idx == 0 and len(logged_examples) < num_examples_to_log:
                    for i in range(min(len(batch['input_ids']), num_examples_to_log)):
                        # Decode input text
                        input_text = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                        
                        # Get raw logits and probabilities
                        true_logit = true_false_logits[i, 0].item()
                        false_logit = true_false_logits[i, 1].item()
                        true_prob = probs[i, 0].item()
                        false_prob = probs[i, 1].item()
                        
                        # Get prediction and confidence
                        pred = "FEASIBLE" if preds[i].item() == 1 else "NOT FEASIBLE"
                        conf = probs[i, 0].item() if pred == "FEASIBLE" else 1 - probs[i, 0].item()
                        true_label = "FEASIBLE" if batch['raw_labels'][i].item() == 1 else "NOT FEASIBLE"
                        
                        logged_examples.append({
                            'input': input_text,
                            'prediction': pred,
                            'confidence': conf,
                            'true_label': true_label,
                            'true_logit': true_logit,
                            'false_logit': false_logit,
                            'true_prob': true_prob,
                            'false_prob': false_prob
                        })
                
                # Clear memory periodically
                if batch_idx % 10 == 0:
                    del outputs, logits, probs, preds, loss
                    torch.cuda.empty_cache()
        
        # Print example predictions
        print("\n📝 Example Predictions:")
        print("=" * 100)
        for ex in logged_examples:
            print(f"\nInput:\n{ex['input']}")
            print(f"\nRaw Model Outputs:")
            print(f"True token  - logit: {ex['true_logit']:6.2f}, probability: {ex['true_prob']:.2%}")
            print(f"False token - logit: {ex['false_logit']:6.2f}, probability: {ex['false_prob']:.2%}")
            print(f"\nFinal Prediction: {ex['prediction']} (confidence: {ex['confidence']:.2%})")
            print(f"True Label: {ex['true_label']}")
            print("=" * 100)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _calculate_metrics(self, labels, preds, probs=None):
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1  # 1 = feasible
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
                auc = roc_auc_score(labels, probs)
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
    parser = argparse.ArgumentParser(description="Train Feasibility Judge")
    parser.add_argument('--data_dir', type=str, default='./data/feasibility',
                       help='Path to prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/feasibility_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,  # Reduced for generation
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,  # Lower for LLM finetuning
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--freeze_layers', type=int, default=20,
                       help='Number of layers to freeze in base model')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--visualize_during_training', action='store_true',
                       help='Create MI visualizations during training')
    parser.add_argument('--visualization_interval', type=int, default=3,
                       help='Create visualizations every N epochs')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'weight_decay': args.weight_decay,
        'patience': 5,
        'output_dir': args.output_dir,
        'use_wandb': args.use_wandb,
        'visualize_during_training': args.visualize_during_training,
        'visualization_interval': args.visualization_interval
    }
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Initialize tokenizer first
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Configure padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize model config
    model_config = FeasibilityJudgeConfig(
        freeze_layers=args.freeze_layers,
        max_length=args.max_length
    )
    
    # Load raw data
    print(f"Loading dataset from {args.data_dir}")
    with open(Path(args.data_dir) / 'train.json', 'r') as f:
        train_data = json.load(f)
    with open(Path(args.data_dir) / 'val.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Initialize model
    model = FeasibilityJudge(model_config)
    
    # Create trainer
    trainer = FeasibilityJudgeTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        config=config
    )
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main() 