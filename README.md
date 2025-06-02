# 🛡️ Guardian-Loop: Micro-Judge Training & Analysis

A lightweight judge system with mechanistic interpretability for LLMs. This guide shows you how to train and evaluate it.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare safety datasets
cd guardian-loop
python src/data/prepare_safety_data.py

# 3. Run the automated pipeline
python guardian-loop/run_full_pipeline.py \
    --visualize-training \
    --skip-open-ended \
    --run-advanced-mi

# 4. Launch the dashboard
streamlit run mi_dashboard.py
```

## Requirements

- Python 3.8+
- CUDA GPU with 16GB+ VRAM (for Llama 3.1 8B)
- 50GB disk space
- Hugging Face account (for model downloads)

## Setup

### 1. Data Preparation

The system uses multiple safety datasets that are automatically downloaded and prepared:

```bash
python src/data/prepare_safety_data.py
```

This downloads and processes RealToxicityPrompts, ToxicChat, JailbreakBench, and Harmbench

Output files:
- `data/prepared/train.json` - Training data
- `data/prepared/val.json` - Validation data
- `data/prepared/test.json` - Test data

Each sample has:
```json
{
  "prompt": "User input text",
  "label": 1,  // 1=safe, 0=unsafe
  "source": "dataset_name"
}
```

### 2. Training the Safety Judge

```bash
python src/train_safety_judge.py \
    --num_epochs 15 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 4 \
    --freeze_layers 20 \
    --visualize_during_training \
    --visualization_interval 1
```

Parameters explained:
- `--freeze_layers`: How many transformer layers to freeze (saves memory)
- `--gradient_accumulation_steps`: Accumulate gradients for larger effective batch
- `--visualize_during_training`: Generate MI visualizations during training
- `--run_advanced_mi`: Enable neuron/circuit analysis (slower)

Default training settings:
- Model: Llama 3.1 8B (frozen first 20 layers)
- Epochs: 15
- Batch size: 4 (with gradient accumulation = 16 effective)
- Learning rate: 2e-4
- Max sequence length: 256 tokens

### 4. Evaluating the Model

After training, evaluate performance:

```bash
python src/evaluate_safety.py \
    --model_path outputs/checkpoints/best_model.pt \
    --test_data data/prepared/test.json \
    --output_dir outputs/evaluation
```


### 5. Advanced MI Analysis

For deep mechanistic interpretability:

```bash
python src/run_mi_analysis.py \
    --model_path outputs/checkpoints/best_model.pt \
    --output_dir outputs/mi_analysis_demo
```

## Pipeline Scripts

### Full Pipeline (No Adversarial)
```bash
python guardian-loop/run_full_pipeline.py \
    --visualize-training \
    --skip-open-ended \
    --run-advanced-mi
```
Runs: data prep → training → evaluation → MI analysis

### Custom Pipeline
```bash
python run_full_pipeline.py \
    --visualize-training \
    --skip-open-ended \
    --num-epochs 20 \
    --batch-size 8
```
