# Mechanistic Interpretability During Training

## Overview

This feature allows you to visualize how your safety judge's internal mechanisms evolve during training. It's particularly useful for understanding:

1. **When** the model learns to distinguish safe vs unsafe prompts
2. **Which layers** become specialized for safety detection
3. **How** the safe/unsafe circuits diverge over epochs

## Usage

### Option 1: Using the training script directly
```bash
./train_optimized.sh --visualize
```

### Option 2: Using the full pipeline
```bash
python run_full_pipeline.py --visualize-training --skip-rainbow
```

### Option 3: Training command with custom interval
```bash
python src/train_safety_judge.py \
    --visualize_during_training \
    --visualization_interval 2  # Create visualizations every 2 epochs
```

## What Gets Created

During training, the following visualizations are generated:

### 1. **Token Attribution Heatmaps** (`epoch_N_sample_X_label_tokens.html`)
- Shows which tokens the model focuses on
- Compare how attention changes between early and late epochs
- See if the model learns to identify safety-critical keywords

### 2. **Layer Activation Patterns** (`epoch_N_sample_X_label_layers.html`)
- Visualizes activation magnitudes across layers
- Shows MLP vs Attention contributions
- Identifies which layers become safety-specialized

### 3. **Safe vs Unsafe Circuit Comparison** (`epoch_N_circuit_comparison.html`)
- Compares neural circuits for safe and unsafe prompts
- Shows embedding divergence and cosine similarity
- Identifies the "critical layer" where safe/unsafe paths diverge

### 4. **Training Progress** (`training_progress.html`)
- Plots accuracy, F1, and loss over epochs
- Updated after each visualization interval
- Helps identify when the model converges

## Interpreting the Visualizations

### Early Training (Epochs 1-3)
- Token attributions are noisy and unfocused
- Layer activations are similar for safe/unsafe
- Circuit divergence is minimal

### Mid Training (Epochs 4-10) 
- Token focus sharpens on safety-critical words
- Specific layers (typically 22-26) show increased activation
- Clear divergence emerges between safe/unsafe circuits

### Late Training (Epochs 11-15)
- Strong, consistent token attribution patterns
- Specialized safety detection layers
- Maximum circuit divergence stabilizes

## Example Insights

From our experiments, we typically observe:

1. **Layer 23-24** become the "safety detection hub"
2. **Attention heads** in layers 20-26 specialize for harmful patterns
3. **Circuit divergence** peaks around layer 24-25
4. The model learns general safety patterns before specific harmful categories

## Performance Considerations

- Visualizations add ~20-30 seconds per epoch
- HTML files are ~200-500KB each
- Total storage: ~50-100MB for a full training run

## Viewing the Results

1. Navigate to the visualizations directory:
   ```bash
   cd outputs/checkpoints/training_visualizations/
   ```

2. Open any HTML file in your browser:
   ```bash
   open epoch_1_sample_0_safe_tokens.html  # macOS
   # or
   xdg-open epoch_1_sample_0_safe_tokens.html  # Linux
   ```

3. Compare visualizations across epochs to see evolution

## Tips

- Use `--visualization_interval 1` for detailed analysis (every epoch)
- Use `--visualization_interval 5` for faster training with key checkpoints
- The circuit comparison is most informative in later epochs
- Token attributions stabilize faster than layer patterns 