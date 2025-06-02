"""
Evaluation script for Guardian-Loop Safety Judge
Evaluates model performance with proper metrics and log probability analysis
"""

import torch
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.safety_judge import SafetyJudge, SafetyJudgeConfig
from mi_tools.advanced_analysis import AdvancedSafetyAnalyzer, create_training_evolution_visualization
from train_safety_judge import SafetyDataset  # Reuse the same dataset class


class SafetyJudgeEvaluator:
    """Comprehensive evaluator for the safety judge model"""
    
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.device = next(model.parameters()).device
        
        # Create dataloader using same approach as training
        self.test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
    def _create_dataloader(self, dataset, shuffle=False):
        """Create a DataLoader identical to training"""
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
                    torch.zeros(pad_len, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
                ]))
                attention_mask.append(torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len)
                ]))
                
                # Pad labels
                labels.append(torch.cat([
                    item['labels'],
                    torch.full((pad_len,), -100, dtype=torch.long)
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
            batch_size=8,  # Same as training
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0
        )
        
    def evaluate_dataset(self):
        """Evaluate model on test dataset using same approach as training validation"""
        
        print(f"Evaluating on {len(self.test_dataset)} samples...")
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        all_log_prob_diffs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Forward pass - identical to training validation
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Get predictions from logits - identical to training
                logits = outputs['logits']
                true_false_logits = logits[:, [self.test_dataset.true_token_id, self.test_dataset.false_token_id]]
                probs = torch.softmax(true_false_logits, dim=-1)
                preds = (probs[:, 0] > 0.5).long()
                
                # Calculate log prob differences for analysis
                log_probs = torch.log_softmax(true_false_logits, dim=-1)
                log_prob_diff = log_probs[:, 0] - log_probs[:, 1]  # log P(True) - log P(False)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 0].cpu().numpy())  # Probability of True/safe
                all_labels.extend(batch['raw_labels'].cpu().numpy())
                all_log_prob_diffs.extend(log_prob_diff.cpu().numpy())
                
                # Clear memory periodically
                if batch_idx % 10 == 0:
                    del outputs, logits, probs, preds, loss
                    torch.cuda.empty_cache()
        
        # Calculate metrics using same method as training
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(self.test_loader)
        
        # Add log probability analysis
        metrics['log_prob_analysis'] = self._analyze_log_probs(
            all_log_prob_diffs, all_labels, all_preds
        )
        
        # Prepare raw results for visualization
        raw_results = {
            'predictions': all_preds,
            'true_labels': all_labels,
            'confidences': all_probs,
            'log_prob_diffs': all_log_prob_diffs,
            'failed_predictions': self._get_failed_predictions(all_labels, all_preds)
        }
        
        return metrics, raw_results
    
    def _calculate_metrics(self, labels, preds, probs=None):
        """Calculate evaluation metrics - identical to training"""
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1  # 1 = safe
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
        
        # Additional metrics for evaluation
        cm = confusion_matrix(labels, preds)
        safe_correct = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        unsafe_correct = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        fnr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        metrics.update({
            'confusion_matrix': cm.tolist(),
            'safe_accuracy': safe_correct,
            'unsafe_accuracy': unsafe_correct,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'avg_confidence': np.mean(probs) if probs is not None else 0,
            'calibration': self._calculate_calibration(labels, preds, probs) if probs is not None else 0
        })
        
        return metrics
    
    def _analyze_log_probs(self, log_prob_diffs, y_true, y_pred):
        """Analyze log probability distributions"""
        
        # Separate by true label
        safe_diffs = [d for d, y in zip(log_prob_diffs, y_true) if y == 1]
        unsafe_diffs = [d for d, y in zip(log_prob_diffs, y_true) if y == 0]
        
        # Separate by correctness
        correct_diffs = [d for d, yt, yp in zip(log_prob_diffs, y_true, y_pred) if yt == yp]
        incorrect_diffs = [d for d, yt, yp in zip(log_prob_diffs, y_true, y_pred) if yt != yp]
        
        return {
            'mean_safe_diff': np.mean(safe_diffs) if safe_diffs else 0,
            'mean_unsafe_diff': np.mean(unsafe_diffs) if unsafe_diffs else 0,
            'std_safe_diff': np.std(safe_diffs) if safe_diffs else 0,
            'std_unsafe_diff': np.std(unsafe_diffs) if unsafe_diffs else 0,
            'mean_correct_diff': np.mean(correct_diffs) if correct_diffs else 0,
            'mean_incorrect_diff': np.mean(incorrect_diffs) if incorrect_diffs else 0,
            'separation_score': abs(np.mean(safe_diffs) - np.mean(unsafe_diffs)) if safe_diffs and unsafe_diffs else 0
        }
    
    def _calculate_calibration(self, y_true, y_pred, confidences, n_bins=10):
        """Calculate expected calibration error (ECE)"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(c >= bin_lower) & (c < bin_upper) for c in confidences]
            prop_in_bin = sum(in_bin) / len(confidences)
            
            if prop_in_bin > 0:
                accuracy_in_bin = sum(y_true[i] == y_pred[i] for i in range(len(y_true)) if in_bin[i]) / sum(in_bin)
                avg_confidence_in_bin = sum(confidences[i] for i in range(len(confidences)) if in_bin[i]) / sum(in_bin)
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _get_failed_predictions(self, true_labels, predictions):
        """Get failed predictions for MI analysis"""
        failed = []
        for i, (true_label, pred) in enumerate(zip(true_labels, predictions)):
            if true_label != pred:
                # Get the original prompt from dataset
                prompt = self.test_dataset.data[i]['prompt']
                failed.append((prompt, true_label, pred))
        return failed
    
    def visualize_results(self, metrics, raw_results, output_dir: Path):
        """Create visualizations of evaluation results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Safe', 'Predicted Unsafe'],
                   yticklabels=['True Safe', 'True Unsafe'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        
        # 2. Log Probability Distribution (if available)
        if raw_results['log_prob_diffs']:
            plt.figure(figsize=(10, 6))
            
            # Separate by true label
            safe_diffs = [d for d, y in zip(raw_results['log_prob_diffs'], 
                                           raw_results['true_labels']) if y == 1]
            unsafe_diffs = [d for d, y in zip(raw_results['log_prob_diffs'], 
                                             raw_results['true_labels']) if y == 0]
            
            plt.hist(safe_diffs, bins=30, alpha=0.5, label='True Safe', color='green')
            plt.hist(unsafe_diffs, bins=30, alpha=0.5, label='True Unsafe', color='red')
            plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
            plt.xlabel('Log P(True) - Log P(False)')
            plt.ylabel('Count')
            plt.title('Log Probability Difference Distribution')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'log_prob_distribution.png')
            plt.close()
        
        # 3. Confidence Calibration Plot
        plt.figure(figsize=(8, 8))
        
        # Create calibration bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = (np.array(raw_results['confidences']) >= bin_boundaries[i]) & \
                      (np.array(raw_results['confidences']) < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(np.array(raw_results['true_labels'])[bin_mask] == 
                                 np.array(raw_results['predictions'])[bin_mask])
                bin_conf = np.mean(np.array(raw_results['confidences'])[bin_mask])
                bin_accs.append(bin_acc)
                bin_confs.append(bin_conf)
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_accs.append(0)
                bin_confs.append(bin_centers[i])
                bin_counts.append(0)
        
        # Plot calibration
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.scatter(bin_confs, bin_accs, s=np.array(bin_counts)*10, alpha=0.7, label='Model')
        plt.xlabel('Mean Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Plot (ECE: {metrics["calibration"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png')
        plt.close()
        
        print(f"📊 Visualizations saved to {output_dir}")


def perform_advanced_mi_analysis(model, tokenizer, test_dataset, test_data, output_dir: Path):
    """Perform advanced mechanistic interpretability analysis"""
    
    print("\nStarting Advanced Mechanistic Interpretability Analysis...")
    
    # Separate safe and unsafe samples
    safe_samples = [item['prompt'] for item in test_data if item['label'] == 1][:50]
    unsafe_samples = [item['prompt'] for item in test_data if item['label'] == 0][:50]
    
    # Create analyzer
    analyzer = AdvancedSafetyAnalyzer(model, tokenizer)
    
    # 1. Identify safety neurons
    safety_neurons = analyzer.identify_safety_neurons(safe_samples, unsafe_samples)
    
    # 2. Trace safety circuits
    safety_circuits = analyzer.trace_safety_circuits()
    
    # 3. Create visualizations
    mi_viz_dir = output_dir / 'mi_analysis'
    mi_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Neuron activation map
    neuron_map = analyzer.create_neuron_activation_map()
    neuron_map.write_html(str(mi_viz_dir / 'neuron_activation_map.html'))
    
    # Circuit diagram
    circuit_diagram = analyzer.create_circuit_diagram()
    circuit_diagram.write_html(str(mi_viz_dir / 'safety_circuits.html'))
    
    # 4. Analyze failure modes
    # Get failed predictions from evaluation - reuse the test_dataset
    evaluator = SafetyJudgeEvaluator(model, tokenizer, test_dataset)
    _, raw_results = evaluator.evaluate_dataset()
    failure_analysis = analyzer.analyze_failure_modes(raw_results['failed_predictions'])
    
    # 5. Export analysis
    analyzer.export_analysis(mi_viz_dir)
    
    # 6. Create training evolution visualization
    evolution_viz = create_training_evolution_visualization(output_dir.parent)
    evolution_viz.write_html(str(mi_viz_dir / 'training_evolution.html'))
    
    print(f"\n✅ Advanced MI Analysis Complete!")
    print(f"   - Found {len(safety_neurons)} safety-critical neurons")
    print(f"   - Identified {len(safety_circuits)} safety detection circuits")
    print(f"   - Analyzed {len(raw_results['failed_predictions'])} failure cases")
    print(f"   - Results saved to {mi_viz_dir}")
    
    return {
        'num_safety_neurons': len(safety_neurons),
        'num_circuits': len(safety_circuits),
        'failure_analysis': failure_analysis
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Guardian-Loop Safety Judge")
    parser.add_argument('--model_path', type=str, 
                       default='./outputs/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='./data/prepared/test.json',
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--skip_mi_analysis', action='store_true',
                       help='Skip advanced MI analysis')
    args = parser.parse_args()
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('model_config', SafetyJudgeConfig())
    
    # Load tokenizer first (same as training)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = SafetyJudge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Create test dataset using same approach as training
    test_dataset = SafetyDataset(test_data, tokenizer, config)
    
    # Create evaluator
    evaluator = SafetyJudgeEvaluator(model, tokenizer, test_dataset)
    
    # Run evaluation
    print("\n🔍 Running evaluation...")
    metrics, raw_results = evaluator.evaluate_dataset()
    
    # Print results
    print("\n📊 Evaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    if metrics['auc'] is not None:
        print(f"  AUC-ROC: {metrics['auc']:.3f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.3f}")
    print(f"  Average Confidence: {metrics['avg_confidence']:.3f}")
    print(f"  Calibration Error (ECE): {metrics['calibration']:.3f}")
    
    if 'log_prob_analysis' in metrics:
        print("\n📈 Log Probability Analysis:")
        analysis = metrics['log_prob_analysis']
        print(f"  Mean log diff (safe): {analysis['mean_safe_diff']:.3f} ± {analysis['std_safe_diff']:.3f}")
        print(f"  Mean log diff (unsafe): {analysis['mean_unsafe_diff']:.3f} ± {analysis['std_unsafe_diff']:.3f}")
        print(f"  Separation score: {analysis['separation_score']:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")
    
    # Create visualizations
    evaluator.visualize_results(metrics, raw_results, output_dir)
    
    # Save detailed predictions for analysis
    predictions_df = pd.DataFrame({
        'prompt': [item['prompt'] for item in test_data],
        'true_label': raw_results['true_labels'],
        'predicted_label': raw_results['predictions'],
        'confidence': raw_results['confidences'],
        'log_prob_diff': raw_results['log_prob_diffs']
    })
    predictions_df.to_csv(output_dir / 'detailed_predictions.csv', index=False)
    
    # Perform advanced MI analysis
    if not args.skip_mi_analysis:
        mi_results = perform_advanced_mi_analysis(model, tokenizer, test_dataset, test_data, output_dir)
        
        # Add MI results to metrics
        metrics['mi_analysis'] = mi_results
        
        # Save updated metrics
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main() 