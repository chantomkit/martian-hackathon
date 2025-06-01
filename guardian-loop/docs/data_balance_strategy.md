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

Synthetic data generation is **reserved for the Open-Ended adversarial phase** where we:
- Generate mutations of unsafe prompts
- Create adversarial examples
- Test edge cases

This separation ensures:
- Baseline model trains on real, high-quality data
- Adversarial improvement is measurable
- We can track which synthetic patterns actually help 

## Current Data Imbalance Issue

The current implementation creates a severely imbalanced dataset:
- **Unsafe prompts**: ~50,000 (from all safety datasets)
- **Safe prompts**: ~5,000 (limited to `train_safe` from Anthropic)

This 10:1 ratio is problematic because:
1. The model will be biased toward predicting "unsafe"
2. High false positive rate on benign prompts
3. Poor generalization to real-world usage (mostly safe prompts)

## Solution: Balanced Dataset Strategy

### 1. Expand Safe Data Sources

```python
def prepare_balanced_dataset(self):
    # Unsafe data (keep as is)
    unsafe_data = self.prepare_all_unsafe_datasets()  # ~50K samples
    
    # Safe data - expand sources
    safe_data = []
    
    # 1. Anthropic safe prompts
    safe_data.extend(self.prepare_anthropic_data()['safe'])  # ~5K
    
    # 2. Common instruction datasets (filtered for safety)
    safe_data.extend(self.prepare_safe_instructions())  # ~20K
    
    # 3. General conversation data
    safe_data.extend(self.prepare_conversation_data())  # ~15K
    
    # 4. Domain-specific safe queries
    safe_data.extend(self.prepare_domain_safe_data())  # ~10K
    
    # Total: ~50K safe samples to match unsafe
```

### 2. Additional Safe Data Sources

#### A. Filtered Instruction Datasets
```python
def prepare_safe_instructions(self):
    """Extract safe prompts from instruction datasets"""
    datasets = [
        "databricks/databricks-dolly-15k",
        "OpenAssistant/oasst1",
        "sahil2801/CodeAlpaca-20k"
    ]
    
    safe_prompts = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        # Filter for clearly safe, helpful prompts
        safe_prompts.extend(self.filter_safe_prompts(dataset))
    
    return safe_prompts
```

#### B. Synthetic Safe Prompts
Synthetic data generation is **reserved for the Open-Ended adversarial phase** where we:
- Generate mutations of unsafe prompts
- Create adversarial examples
- Test edge cases

For the balanced training set, we use only real data.

#### C. Domain-Specific Safe Queries
```python
def prepare_domain_safe_data(self):
    """Common safe queries by domain"""
    domains = {
        'education': [
            "Explain photosynthesis",
            "How do I solve quadratic equations?",
            "What caused World War I?"
        ],
        'cooking': [
            "Recipe for chocolate chip cookies",
            "How to make pasta from scratch",
            "Best way to cook salmon"
        ],
        'technology': [
            "How to use Git for version control",
            "Explain machine learning basics",
            "What is cloud computing?"
        ],
        'health': [
            "Benefits of regular exercise",
            "How to maintain good posture",
            "Importance of sleep hygiene"
        ]
    }
    # Expand with variations
```

### 3. Balanced Sampling Strategy

```python
def create_balanced_splits(self, safe_data, unsafe_data):
    """Create balanced train/val splits"""
    
    # Target: 50/50 safe/unsafe ratio
    n_samples = min(len(safe_data), len(unsafe_data))
    
    # Sample equally from both
    safe_sampled = random.sample(safe_data, n_samples)
    unsafe_sampled = random.sample(unsafe_data, n_samples)
    
    # Combine and shuffle
    all_data = safe_sampled + unsafe_sampled
    random.shuffle(all_data)
    
    # Split 90/10 for train/val
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    return train_data, val_data
```

### 4. Implementation Plan

1. **Immediate fix** (for hackathon):
   ```python
   # In prepare_safety_data.py
   safe_data = self.prepare_anthropic_data()['safe']
   # Duplicate safe samples to balance
   safe_data = safe_data * 10  # Quick hack to get ~50K
   ```

2. **Proper solution** (post-hackathon):
   - Implement all safe data sources above
   - Add data quality filters
   - Ensure diversity in safe prompts

### 5. Expected Outcomes

With balanced data:
- **Accuracy**: 85-90% (currently might be inflated)
- **False Positive Rate**: <10% (currently likely >30%)
- **F1 Score**: 0.85+ for both classes
- **Better real-world performance**

### 6. Validation Strategy

Test on realistic distribution:
```python
# Real-world distribution: 90% safe, 10% unsafe
realistic_test_set = {
    'safe': 900,
    'unsafe': 100
}
```

This ensures our model performs well in actual deployment scenarios. 