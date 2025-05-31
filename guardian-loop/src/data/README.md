# Data Preparation Documentation

## Overview

This directory contains scripts for preparing safety datasets for training the Guardian-Loop safety judge.

## Dataset Information

### Prompt-Only Datasets (Recommended)

These datasets label individual user prompts for toxicity:

1. **ToxicChat** (lmsys/toxic-chat)
   - 10K real user prompts from ChatGPT conversations
   - Binary toxicity labels (0=non-toxic, 1=toxic)
   - Includes jailbreaking labels
   - **✅ Perfect for prompt-only detection**

2. **Real-Toxicity-Prompts** (allenai/real-toxicity-prompts)
   - 99K prompts with continuous toxicity scores
   - Multiple toxicity dimensions (threat, insult, profanity, etc.)
   - **✅ Perfect for prompt-only detection**

3. **JailbreakBench**
   - Curated collection of jailbreak attempts
   - All prompts are unsafe by design
   - **✅ Perfect for adversarial prompt detection**

### QA-Pair Datasets (Limited Use)

These datasets label entire conversations or QA pairs:

1. **BeaverTails** (PKU-Alignment/BeaverTails)
   - ⚠️ **WARNING**: Labels entire question-answer pairs, NOT individual prompts
   - A harmful prompt might be labeled "safe" if the response mitigates the harm
   - We use category labels as a proxy, but this is an approximation
   - **⚠️ Use with caution for prompt-only detection**

2. **WildChat** (allenai/WildChat-1M)
   - Labels entire conversations for toxicity
   - We extract first user message as prompt
   - **⚠️ Limited suitability for prompt-only detection**

## Running Data Preparation

```bash
# Prepare datasets with default settings
python -m src.data.prepare_safety_data

# Custom settings
python prepare_safety_data.py --balance-ratio 0.6 --use-mutations
```

## Dataset Statistics

After preparation, you'll get:
- Train: ~70% of data
- Validation: ~10% of data  
- Test: ~20% of data

The script prioritizes prompt-only datasets and aims for a 50/50 balance of safe/unsafe prompts.

## Important Notes

1. **BeaverTails Limitation**: This popular dataset is designed for QA-moderation, not prompt-only toxicity detection. Its labels reflect whether the entire interaction is harmful, not whether the prompt alone is toxic.

2. **Mutation Strategy**: The script can generate "safe" versions of unsafe prompts using GPT-4o to help the model learn boundaries.

3. **Error Handling**: If Real-Toxicity-Prompts fails to load, check that toxicity scores are properly handled (some may be None). 