"""
Training script for the Safety Judge
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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
        
        # Initialize MI visualizer if requested
        self.mi_visualizer = None
        if config.get('visualize_during_training', False):
            from src.mi_tools.visualization import SafetyJudgeMIVisualizer
            self.mi_visualizer = SafetyJudgeMIVisualizer(model, tokenizer)
            self.visualization_dir = Path(config['output_dir']) / 'training_visualizations'
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            print(f"📊 MI visualizations will be saved to {self.visualization_dir}")
        
        # Get trainable parameters - both probe head and unfrozen layers
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Training: {name}")
        
        print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Use different learning rates for different parts
        optimizer_groups = [
            # Probe head - higher learning rate
            {
                'params': self.model.probe_head.parameters(),
                'lr': config['learning_rate']
            },
            # Unfrozen transformer layers - lower learning rate
            {
                'params': [p for n, p in self.model.base_model.named_parameters() if p.requires_grad],
                'lr': config['learning_rate'] * 0.1  # 10x smaller for pretrained layers
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),  # Standard AdamW betas
            eps=1e-8
        )
        
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        
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
        
        # Gradient accumulation for effective larger batch size
        accumulation_steps = self.config.get('gradient_accumulation_steps', 2)
        
        # Mixed precision training for efficiency
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, labels)
            else:
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * accumulation_steps
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
        
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
            
            # Create MI visualizations every N epochs or on first/last epoch
            visualization_interval = self.config.get('visualization_interval', 3)
            if (self.mi_visualizer and 
                (epoch == 0 or 
                 epoch == self.config['num_epochs'] - 1 or 
                 (epoch + 1) % visualization_interval == 0)):
                self.create_mi_visualizations(epoch + 1, val_metrics)
            
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
        
        if self.mi_visualizer:
            print(f"\n📊 MI visualizations saved to: {self.visualization_dir}")
            print("   View them by opening the HTML files in your browser")
    
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
    
    def create_mi_visualizations(self, epoch: int, val_metrics: Dict):
        """Create MI visualizations for validation samples"""
        if self.mi_visualizer is None:
            return
            
        print(f"\n🔬 Creating MI visualizations for epoch {epoch}...")
        
        # Select a few diverse validation samples
        safe_samples = [s for s in self.val_dataset if s['label'] == 1][:2]
        unsafe_samples = [s for s in self.val_dataset if s['label'] == 0][:2]
        samples = safe_samples + unsafe_samples
        
        # Create visualizations for each sample
        for i, sample in enumerate(samples):
            prompt = sample['prompt']
            label = "safe" if sample['label'] == 1 else "unsafe"
            
            # Token attribution
            token_fig, _ = self.mi_visualizer.create_token_attribution_heatmap(prompt, return_data=True)
            token_path = self.visualization_dir / f"epoch_{epoch}_sample_{i}_{label}_tokens.html"
            token_fig.write_html(str(token_path))
            
            # Layer activations
            layer_fig = self.mi_visualizer.visualize_layer_activations(prompt)
            layer_path = self.visualization_dir / f"epoch_{epoch}_sample_{i}_{label}_layers.html"
            layer_fig.write_html(str(layer_path))
        
        # Compare safe vs unsafe circuits if we have both
        if safe_samples and unsafe_samples:
            circuit_fig, circuit_data = self.mi_visualizer.compare_safe_unsafe_circuits(
                safe_samples[0]['prompt'], 
                unsafe_samples[0]['prompt']
            )
            circuit_path = self.visualization_dir / f"epoch_{epoch}_circuit_comparison.html"
            circuit_fig.write_html(str(circuit_path))
            
            # Log the critical divergence layer
            print(f"   Critical divergence at layer {circuit_data['critical_layer']}")
        
        # Create summary visualization showing metrics evolution
        self._create_metrics_summary(epoch, val_metrics)
        
        print(f"   ✅ Visualizations saved to {self.visualization_dir}")
    
    def _create_metrics_summary(self, epoch: int, val_metrics: Dict):
        """Create a summary plot of training progress"""
        import plotly.graph_objects as go
        
        # Track metrics history
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {'epochs': [], 'accuracy': [], 'f1': [], 'loss': []}
        
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['accuracy'].append(val_metrics['accuracy'])
        self.metrics_history['f1'].append(val_metrics['f1'])
        self.metrics_history['loss'].append(val_metrics['loss'])
        
        # Create plot
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=self.metrics_history['epochs'],
            y=self.metrics_history['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.metrics_history['epochs'],
            y=self.metrics_history['f1'],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='green')
        ))
        
        # Add loss on secondary y-axis
        fig.add_trace(go.Scatter(
            x=self.metrics_history['epochs'],
            y=self.metrics_history['loss'],
            mode='lines+markers',
            name='Loss',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Metric Value',
            yaxis2=dict(
                title='Loss',
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        # Save
        metrics_path = self.visualization_dir / 'training_progress.html'
        fig.write_html(str(metrics_path))


def main():
    parser = argparse.ArgumentParser(description="Train Safety Judge")
    parser.add_argument('--data_dir', type=str, default='./data/prepared/safety_dataset',
                       help='Path to prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,  # Reduced for Llama 3.1
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch = batch_size * this)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,  # Lower for stability
                       help='Learning rate for probe head')
    parser.add_argument('--num_epochs', type=int, default=15,  # More epochs for 8K samples
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--freeze_layers', type=int, default=20,  # Unfreeze last 12 layers
                       help='Number of layers to freeze in base model')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'cls'],
                       help='Pooling strategy for hidden states')
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
        'weight_decay': 0.1,  # Higher weight decay for regularization
        'patience': 5,  # More patience for convergence
        'output_dir': args.output_dir,
        'use_wandb': args.use_wandb,
        'visualize_during_training': args.visualize_during_training,
        'visualization_interval': args.visualization_interval
    }
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Load dataset - use JSON files directly
    import json
    from torch.utils.data import Dataset
    
    class SafetyDataset(Dataset):
        def __init__(self, file_path):
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    print(f"Loading dataset from {args.data_dir}")
    train_dataset = SafetyDataset('./data/prepared/train.json')
    val_dataset = SafetyDataset('./data/prepared/val.json')
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model_config = SafetyJudgeConfig(
        freeze_layers=args.freeze_layers,
        max_length=args.max_length,
        use_pooler=args.pooling,
        probe_hidden_size=128,  # Smaller for 8K dataset
        dropout_rate=0.2  # Higher dropout
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main() 