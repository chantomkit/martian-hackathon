# Mini-Judge Model Selection Guide

## Overview
This document outlines the selection criteria and recommended models for implementing mini-judges in our system. Mini-judges are smaller LLMs used to evaluate model outputs for safety, factuality, bias, and other quality metrics.

## Base Model Selection
Our primary choice is LLaMA 3.1 (Meta-Llama-3-8B-Instruct), which offers:
- State-of-the-art performance for its size
- Excellent instruction-following capabilities
- Good balance of performance and resource efficiency

## Alternative Models (≤13B)

### 1. Mistral-7B-Instruct
- **Source**: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Key Features**:
  - Optimized for chat and instruction following
  - Strong factuality performance
  - Efficient resource utilization

### 2. Nous-Hermes-2 Mistral 7B
- **Source**: [NousResearch/Nous-Hermes-2-Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)
- **Key Features**:
  - Safety-focused fine-tuning
  - Excellent at nuanced scoring
  - Enhanced output safety

### 3. OpenHermes 2.5 Mistral 7B
- **Source**: [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
- **Key Features**:
  - Strong reasoning capabilities
  - Well-suited for instruction-based tasks
  - Good benchmark performance

### 4. MythoMax-L2 13B
- **Source**: [Gryphe/MythoMax-L2-13B](https://huggingface.co/Gryphe/MythoMax-L2-13B)
- **Key Features**:
  - LLaMA 2-based architecture
  - Strong reasoning capabilities
  - Alignment-tuned for safety tasks

### 5. Dolphin 2.6 Mistral 7B
- **Source**: [cognitivecomputations/dolphin-2.6-mistral-7b-dpo](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo)
- **Key Features**:
  - Dialogue and safety optimization
  - Built-in refusal strategies
  - Good balance of performance and safety

## Specialized Evaluation Models

### Evaluation-Specific Models
1. **Bactrian-X / AlpacaEval**
   - **Source**: [declare-lab/flan-alpaca-gpt4-eval](https://huggingface.co/declare-lab/flan-alpaca-gpt4-eval)
   - Purpose: Specifically designed for response evaluation

2. **GPT4All (Quantized)**
   - **Source**: [nomic-ai/gpt4all-13b-snoozy](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy)
   - Purpose: Lightweight evaluation with minimal resource requirements

## Implementation Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_judge_model(model_id: str):
    """
    Load a judge model from Hugging Face.
    
    Args:
        model_id (str): The Hugging Face model ID
        
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    return model, tokenizer
```

## Selection Criteria
When choosing a model for specific judging tasks, consider:
1. Task requirements (safety, factuality, bias detection)
2. Resource constraints
3. Latency requirements
4. Specific domain expertise needed

## Recommended Configurations
- **Safety Evaluation**: Nous-Hermes-2 or Dolphin 2.6
- **Factuality Checking**: Mistral-7B-Instruct
- **General Purpose**: LLaMA 3.1 or OpenHermes 2.5
- **Resource-Constrained**: GPT4All quantized versions 