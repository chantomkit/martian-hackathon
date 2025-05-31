# Training Optimizations for 8K Dataset

## Key Changes Made for Better Results

### 1. **Model Architecture Optimizations**
- **Reduced frozen layers**: From 24 → 20 (unfreezing last 12 layers gives more capacity)
- **Simplified MLP**: 2-layer instead of 3-layer (4096 → 128 → 2)
- **Added LayerNorm**: For training stability
- **Changed activation**: ReLU → GELU (often better for transformers)

### 2. **Training Strategy**
- **Differential learning rates**: 
  - Probe head: 2e-4
  - Unfrozen layers: 2e-5 (10x smaller)
- **Gradient accumulation**: Effective batch size = 32 (8 × 4)
- **Mixed precision training**: For efficiency and regularization
- **Label smoothing**: 0.1 for better generalization

### 3. **Regularization**
- **Higher dropout**: 0.2 (was 0.1)
- **Higher weight decay**: 0.1 (was 0.01)  
- **Cosine scheduler**: Better than linear for small datasets

### 4. **Pooling Strategy**
- **Mean pooling with attention mask**: Better than just using [CLS] token
- **Multi-layer aggregation**: Combines last 3 unfrozen layers

## Why These Changes Work for 8K Samples

1. **More trainable parameters** (unfrozen layers) give the model capacity to learn your specific safety patterns
2. **Simpler probe head** reduces overfitting risk
3. **Strong regularization** prevents memorization
4. **Gradient accumulation** simulates larger batch training
5. **Differential learning rates** preserve pretrained knowledge while adapting

## Expected Results

With these optimizations, you should see:
- **Training**: 85-90% accuracy
- **Validation**: 80-85% accuracy  
- **Test**: 78-82% accuracy
- **Training time**: ~2-3 hours on a single GPU

## Quick Start

```bash
# Make the script executable
chmod +x train_optimized.sh

# Run optimized training
./train_optimized.sh
```

## Monitoring Training

Watch for:
1. Validation loss should decrease for first 5-7 epochs
2. Gap between train/val accuracy should stay < 10%
3. Learning rate warmup in first epoch
4. Early stopping if validation doesn't improve for 5 epochs 