# Data Balance Strategy for Guardian-Loop Safety Judge

## Overview

For effective safety classification, we need careful balance between safe and unsafe prompts across all dataset splits. Here's our approach:

## Dataset Composition

### Real Datasets Only (No Synthetic Generation in Initial Training)

**Unsafe Sources:**
- **ToxicChat** (3k samples): Real user-AI conversations with toxicity labels
- **PKU-SafeRLHF** (3k samples): Safety preference data with unsafe examples  
- **Anthropic Red Team** (2k samples): Real adversarial attempts

**Safe Sources:**
- **HH-RLHF** (5k samples): Human preference data (chosen responses)
- **OpenAssistant** (3k samples): Helpful, harmless conversations
- **Stanford Alpaca** (fallback): General instruction-following prompts

Total: ~16k samples before balancing

## Balance Ratios

### Training Set (50/50 Default)
- **Rationale**: Equal representation prevents the model from simply memorizing class imbalance
- **Benefits**: 
  - Model learns both patterns equally well
  - Reduces bias toward predicting majority class
  - Better calibration of confidence scores

### Validation & Test Sets (50/50 Default)
- **Rationale**: Balanced evaluation gives clear picture of performance on both classes
- **Metrics**: With balanced sets, accuracy ≈ (precision + recall) / 2
- **Alternative**: Can adjust to match real-world distribution if known

## Customizable Balance

The `prepare_dataset()` function supports custom ratios:

```python
dataset = preparer.prepare_dataset(
    train_balance_ratio=0.5,    # 50% safe in training
    eval_balance_ratio=0.4      # 40% safe in val/test (more unsafe)
)
```

## Why This Approach?

1. **No Synthetic Data Initially**: Real examples provide authentic patterns that synthetic data might miss
2. **Balanced Training**: Prevents the trivial solution of always predicting one class
3. **Consistent Evaluation**: Fair assessment of both false positives and false negatives
4. **Flexibility**: Can adjust ratios based on deployment requirements

## Considerations for Production

In real deployment, you might want to:
- Measure actual distribution of queries
- Adjust test set to match production distribution
- Use cost-sensitive learning if false negatives are more costly
- Consider threshold tuning based on business requirements

## Synthetic Data Strategy

Synthetic data generation is **reserved for the Rainbow adversarial phase** where we:
- Generate targeted adversarial examples
- Fill gaps discovered by the MAP-Elites archive
- Create specific attack patterns the model struggles with

This separation ensures:
- Baseline model trains on real, high-quality data
- Adversarial improvement is measurable
- We can track which synthetic patterns actually help 