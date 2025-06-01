# 🚀 Guardian-Loop Quickstart

Get up and running with Guardian-Loop in 5 minutes!

## Prerequisites

```bash
# Python 3.10+
python --version

# CUDA-capable GPU (optional but recommended)
nvidia-smi
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/guardian-loop.git
cd guardian-loop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - HF_TOKEN (required for LLaMA model access)
# - MARTIAN_API_KEY (optional for Martian integration)
```

## Quick Test

```bash
# Test basic functionality
python test_basic_pipeline.py
```

## Full Pipeline

### Option 1: Complete Pipeline (Recommended)

```bash
# Run everything: data prep, training, evaluation, MI analysis
python run_full_pipeline.py

# With visualization during training (slower but insightful)
python run_full_pipeline.py --visualize-training

# Skip Open-Ended adversarial testing (faster)
python run_full_pipeline.py --skip-open-ended
```

### Option 2: Step by Step

```bash
# 1. Prepare data
python src/data/prepare_safety_data.py

# 2. Train safety judge
python src/train_safety_judge.py --num_epochs 3 --batch_size 4

# 3. Evaluate model
python src/evaluate_safety.py

# 4. Generate MI visualizations
python src/mi_tools/visualization.py

# 5. (Optional) Run Open-Ended adversarial testing
python src/adversarial/open_ended_loop.py --iterations 100
```

## Interactive Demo

### Streamlit Dashboard (Recommended)
```bash
streamlit run mi_dashboard.py
# Open http://localhost:8501
```

### Simple Demo
```bash
streamlit run demo.py
# Or CLI mode: python demo.py --cli
```

## Training Dashboard

For real-time training monitoring:
```bash
streamlit run mi_dashboard.py
```

Features:
- Live training metrics
- MI visualizations per epoch
- Test individual prompts
- Open-Ended adversarial testing interface

## Key Files

- `mi_dashboard.py` - Main training & visualization dashboard
- `src/train_safety_judge.py` - Model training script  
- `src/evaluate_safety.py` - Evaluation metrics
- `src/mi_tools/visualization.py` - MI analysis tools
- `src/adversarial/open_ended_loop.py` - Adversarial testing

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python src/train_safety_judge.py --batch_size 2 --gradient_accumulation_steps 8
```

### No GPU Available
```bash
# Force CPU usage (very slow)
export CUDA_VISIBLE_DEVICES=""
python run_full_pipeline.py
```

### Missing API Keys
- HF_TOKEN: Get from https://huggingface.co/settings/tokens
- MARTIAN_API_KEY: Get from https://withmartian.com

## Next Steps

1. **Explore the Dashboard**: Run `streamlit run mi_dashboard.py` for the full experience
2. **Read the Docs**: Check `docs/` for detailed explanations
3. **Customize Training**: Modify hyperparameters in the dashboard
4. **Run Open-Ended**: Test model robustness with adversarial attacks

## Support

- Issues: https://github.com/your-username/guardian-loop/issues
- Documentation: `docs/README.md`
- Paper: `resources/guardian-loop-paper.pdf`

Happy judging! 🛡️

## 📊 Dataset Details

**Total samples**: ~10,000 balanced (targeting ~5K unsafe prompts before mutations)

| Dataset | Samples | Description | Notes |
|---------|---------|-------------|-------|
| Real-Toxicity-Prompts | 4,000 | AllenAI's gold standard with detailed toxicity scores | ✅ Best quality labels |
| HarmBench | 1,000 | Standardized harmful behavior prompts | ✅ Well-categorized |
| BeaverTails | 1,500 | Uses category labels (any True = unsafe) | ✅ Simple labeling |
| ToxicChat | 1,000 | Only jailbreaks & high toxicity | ✅ Conservative approach |
| WildChat-1M | 500 | Real ChatGPT conversations with toxicity labels | ✅ Real-world data |
| JailbreakBench | 500 | Curated jailbreak prompts | ✅ All harmful |
| Mutations | ~500 | Safe versions via Martian API | Optional with API key |

**Dataset Mix Strategy**:
- Target: ~5K unsafe prompts before mutations
- Real-Toxicity-Prompts provides both safe and unsafe based on toxicity scores
- HarmBench, JailbreakBench are 100% unsafe (jailbreak attempts)
- BeaverTails uses category labels (any harmful category = unsafe)
- WildChat-1M uses toxic/redacted flags
- ToxicChat uses jailbreak flag and toxicity scores
- NO keyword-based labeling - only using dataset labels

**Key Features of Real-Toxicity-Prompts**:
- 99K prompts with fine-grained toxicity scores
- Multi-dimensional safety metrics: toxicity, threat, profanity, identity attack, etc.
- Each score ranges from 0.0 to 1.0
- Thresholds used: toxicity>0.3, threat>0.3, severe_toxicity>0.2, etc.

**Why these datasets?**
- **Real labels**: Using only the labels provided by datasets
- **No synthetic labeling**: No keyword matching or heuristics
- **Research quality**: Peer-reviewed, published datasets
- **Clear labeling**: Each dataset has explicit safety/toxicity labels

## 🔬 What Gets Evaluated

1. **Performance Metrics**:
   - Accuracy, Precision, Recall, F1
   - AUC-ROC score
   - Per-category performance
   - Confusion matrix

2. **Mechanistic Interpretability**:
   - Token attribution heatmaps
   - Attention pattern analysis
   - Layer activation visualization
   - Safety pattern detection

3. **Martian Integration**:
   - API response comparison
   - Cost savings calculation
   - Latency measurements
   - Agreement rates

## 💡 Tips

- **No API Key?** The system works without Martian API:
  - Uses template-based mutations (less sophisticated)
  - Runs Martian integration in mock mode
  - Still fully functional for training/evaluation

- **Low on memory?** Reduce batch size:
  ```bash
  python src/train_safety_judge.py --batch_size 8
  ```

- **Want faster training?** Reduce epochs:
  ```bash
  python src/train_safety_judge.py --epochs 1
  ```

- **Testing mutations?** Run the mutation quality test:
  ```bash
  python test_mutation_quality.py
  ```

## 🎯 Expected Results

With the optimized dataset:
- **Accuracy**: 85-90%
- **AUC**: 0.91-0.93
- **False Positive Rate**: <10%
- **Training time**: ~15 min (GPU)
- **Inference latency**: <10ms

## 🛠️ Troubleshooting

```bash
# If you get import errors:
pip install -r requirements.txt

# If you get CUDA errors:
export CUDA_VISIBLE_DEVICES=""  # Force CPU

# If you get memory errors:
python src/train_safety_judge.py --batch_size 4 --gradient_accumulation_steps 4

# Check your setup:
python test_basic_pipeline.py
```

## 📁 Output Locations

- **Prepared data**: `data/prepared/`
- **Trained model**: `models/safety_judge/`
- **Evaluation results**: `outputs/evaluation/`
- **MI visualizations**: `outputs/mi_visualizations/`
- **Logs**: `logs/`

## 🚨 Adding Open-Ended Adversarial (Later)

When you're ready to add adversarial training:
```bash
# Run adversarial discovery
python src/adversarial/open_ended_loop.py --iterations 100

# Retrain with discovered adversarial examples
python src/train_safety_judge.py --include_adversarial
``` 