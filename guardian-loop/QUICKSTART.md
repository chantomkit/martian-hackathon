# Guardian-Loop Quick Start Guide

## 🚀 Running the Full Pipeline

### Option 1: Run Everything (Recommended)
```bash
# Make sure you're in the guardian-loop directory
cd guardian-loop

# Set your Martian API key (optional but recommended)
export MARTIAN_API_KEY='your-key-here'

# Run the complete pipeline
python run_full_pipeline.py
```

This will:
1. Prepare datasets (~5K samples → ~10K with mutations)
2. Train the safety judge (3 epochs)
3. Evaluate performance
4. Generate MI visualizations
5. Launch interactive demo

**Estimated time**: 15-20 minutes with GPU, 1-1.5 hours with CPU

### Option 2: Run Steps Individually

```bash
# Step 1: Prepare data (5-10 min without API, 15-20 min with GPT-4o mutations)
python src/data/prepare_safety_data.py

# Step 2: Train model (20-30 min GPU, 1-2 hours CPU)
python src/train_safety_judge.py --epochs 3 --batch_size 16

# Step 3: Evaluate (5 min)
python src/evaluate_safety.py

# Step 4: Generate MI visualizations (5-10 min)
python src/mi_tools/visualization.py --mode all

# Step 5: Launch demo
streamlit run demo.py
# OR for CLI version:
python demo.py --mode cli
```

### Option 3: Quick Testing

```bash
# If you've already prepared data and trained:
python run_full_pipeline.py --skip-data-prep --skip-training

# Just run the demo:
python run_full_pipeline.py --demo-only

# Just run evaluation:
python run_full_pipeline.py --eval-only
```

## 📊 Dataset Details

**Total samples**: ~5,000-8,000 (balanced towards safe/unsafe)

| Dataset | Samples | Description | Notes |
|---------|---------|-------------|-------|
| WildGuard | 1,000 | State-of-the-art safety dataset | Requires HF login* |
| BeaverTails | 2,000-3,000 | 14 harm categories | Works out of box |
| ToxicChat | 2,000-3,000 | Real toxic conversations | Works out of box |
| ~~WildChat~~ | ~~1,000~~ | ~~Toxic ChatGPT interactions~~ | Skipped (too large) |
| Anthropic HH | 2,000 | Human preference data | Works out of box |
| Mutations | ~100-2,500 | Safe versions of unsafe prompts | Depends on API |

*To use WildGuard dataset:
```bash
# Option 1: Set environment variable
export HF_TOKEN='your-huggingface-token'

# Option 2: Use HF CLI
huggingface-cli login

# Get token from: https://huggingface.co/settings/tokens
```

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

## 🚨 Adding Rainbow Adversarial (Later)

When you're ready to add adversarial training:
```bash
# Run adversarial discovery
python src/adversarial/rainbow_loop.py --iterations 100

# Retrain with discovered adversarial examples
python src/train_safety_judge.py --include_adversarial
``` 