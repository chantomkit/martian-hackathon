# MI Visualizations During Training

Enable mechanistic interpretability visualizations during model training to understand how the safety judge evolves.

## Quick Start

```bash
# Enable visualizations during training
python run_full_pipeline.py --visualize-training

# With specific interval (default: every 3 epochs)
python src/train_safety_judge.py --visualize_during_training --visualization_interval 1

# With advanced MI analysis (slower but more detailed)
python src/train_safety_judge.py --visualize_during_training --run_advanced_mi
```

## Dashboard Usage

```bash
# Run the interactive dashboard
streamlit run mi_dashboard.py

# Then enable visualizations in the training config
```

## Command Line Options

### Basic Pipeline with Visualizations
```bash
python run_full_pipeline.py --visualize-training --skip-open-ended
```

### Standalone Training with MI
```bash
python src/train_safety_judge.py \
  --num_epochs 15 \
  --batch_size 8 \
  --visualize_during_training \
  --visualization_interval 1 \
  --output_dir outputs/checkpoints
```

### Advanced MI Analysis
```bash
# Includes neuron identification and circuit tracing
python src/train_safety_judge.py \
  --visualize_during_training \
  --run_advanced_mi \
  --visualization_interval 5
```

## Visualization Types

### 1. Token Attribution Heatmaps
- Shows which tokens influence safety classification
- Generated for 2 safe and 2 unsafe validation samples
- Saved as: `epoch_{N}_sample_{i}_{label}_tokens.html`

### 2. Layer Activation Patterns
- Visualizes activation magnitudes across transformer layers
- Shows frozen vs fine-tuned layer boundary
- Saved as: `epoch_{N}_sample_{i}_{label}_layers.html`

### 3. Circuit Comparison
- Compares information flow for safe vs unsafe prompts
- Identifies critical divergence layers
- Saved as: `epoch_{N}_circuit_comparison.html`

### 4. Neuron Maps (Advanced MI)
- Identifies safety-specialized neurons
- Shows activation patterns across layers
- Saved as: `epoch_{N}_neuron_map.html`

### 5. Circuit Diagrams (Advanced MI)
- Maps information flow through safety detection circuits
- Shows connected neuron pathways
- Saved as: `epoch_{N}_circuits.html`

### 6. Training Progress
- Metrics evolution across epochs
- Saved as: `training_progress.html`

## Output Structure

```
outputs/checkpoints/training_visualizations/
├── epoch_1_sample_0_safe_tokens.html
├── epoch_1_sample_0_safe_layers.html
├── epoch_1_sample_1_safe_tokens.html
├── epoch_1_sample_1_safe_layers.html
├── epoch_1_sample_2_unsafe_tokens.html
├── epoch_1_sample_2_unsafe_layers.html
├── epoch_1_sample_3_unsafe_tokens.html
├── epoch_1_sample_3_unsafe_layers.html
├── epoch_1_circuit_comparison.html
├── epoch_1_metrics.json
├── epoch_1_neuron_map.html      # If advanced MI enabled
├── epoch_1_circuits.html        # If advanced MI enabled
└── training_progress.html
```

## Performance Considerations

### Time Overhead
- Basic visualizations: +2-3 minutes per epoch
- Advanced MI analysis: +5-10 minutes per epoch

### Memory Usage
- Basic: Minimal additional memory
- Advanced MI: +2-4GB GPU memory for neuron analysis

### Recommended Settings
```bash
# For quick training with basic insights
--visualize_during_training --visualization_interval 3

# For detailed analysis (fewer epochs)
--visualize_during_training --run_advanced_mi --visualization_interval 1 --num_epochs 5

# For production training with periodic checks
--visualize_during_training --visualization_interval 10
```

## Interpreting Results

### Token Attribution Evolution
- Early epochs: Scattered attention, learning basic patterns
- Mid training: Focus on key safety-related tokens
- Late epochs: Refined, consistent attribution patterns

### Layer Activation Changes
- Watch for activation spikes after the frozen layer boundary
- Increasing activation in later layers indicates safety-specific learning
- Stable patterns suggest convergence

### Circuit Comparison Insights
- Divergence point moves earlier as training progresses
- Larger divergence = better safety/unsafe discrimination
- Critical layers stabilize after good training

### Neuron Specialization (Advanced)
- Safety neurons emerge in layers 20-26
- Clusters indicate robust detection mechanisms
- Sparse activations suggest overfitting

## Troubleshooting

### Visualizations Not Generated
- Check output directory permissions
- Ensure Plotly is installed: `pip install plotly`
- Verify CUDA memory if using advanced MI

### HTML Files Not Loading
- Use a modern browser (Chrome/Firefox recommended)
- Check for JavaScript errors in console
- Try opening files directly in browser

### Memory Errors with Advanced MI
- Reduce batch size
- Increase visualization interval
- Disable advanced MI for some epochs

## Integration with Dashboard

The MI dashboard automatically loads and displays these visualizations:

1. Run training with visualizations enabled
2. Open dashboard: `streamlit run mi_dashboard.py`
3. Navigate to "Training Visualizations" section
4. Select epoch with slider
5. View all visualizations in organized tabs

## Best Practices

1. **Start Simple**: Use basic visualizations first
2. **Strategic Intervals**: Visualize more frequently early in training
3. **Compare Epochs**: Look for evolution patterns
4. **Focus on Changes**: Sudden shifts indicate learning milestones
5. **Validate Insights**: Cross-reference with validation metrics

## Example Workflow

```bash
# 1. Quick test with frequent visualizations
python src/train_safety_judge.py --num_epochs 3 --visualize_during_training --visualization_interval 1

# 2. Review in dashboard
streamlit run mi_dashboard.py

# 3. Full training with periodic checks
python src/train_safety_judge.py --num_epochs 15 --visualize_during_training --visualization_interval 5

# 4. Deep dive on interesting epochs
python src/train_safety_judge.py --num_epochs 20 --visualize_during_training --run_advanced_mi --visualization_interval 10
``` 