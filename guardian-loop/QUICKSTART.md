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

## 🚨 Adding Rainbow Adversarial (Later)

When you're ready to add adversarial training:
```bash
# Run adversarial discovery
python src/adversarial/rainbow_loop.py --iterations 100

# Retrain with discovered adversarial examples
python src/train_safety_judge.py --include_adversarial
``` 