# Guardian-Loop Dashboard Guide

## 🚀 Quick Start

```bash
# Make sure you're in the project root
streamlit run guardian-loop/mi_dashboard.py
```

## 📚 Dashboard Sections

### 1. Training Configuration

#### Data & Training Parameters
- **Number of Epochs** (default: 15): How many times to iterate through the training data
- **Batch Size** (default: 4): Number of samples processed together (lower = less GPU memory)
- **Gradient Accumulation** (default: 4): Accumulate gradients over N batches
  - Effective batch size = Batch Size × Gradient Accumulation = 16

#### Optimization Settings
- **Learning Rate** (default: 2e-4): Step size for weight updates
- **Weight Decay** (default: 0.01): L2 regularization to prevent overfitting
- **Freeze Layers** (default: 20): Keep first 20 layers of LLaMA frozen, only train top layers

#### Advanced Options
- **Max Sequence Length** (default: 256): Maximum token length for inputs
- **Early Stopping Patience** (default: 5): Stop if no improvement for 5 epochs
- **Use Weights & Biases**: Enable experiment tracking (requires `wandb login`)
- **Advanced MI Analysis**: Enable neuron/circuit analysis (slower but more detailed)

### 2. Training Visualizations

Visualizations are generated **every epoch** automatically:

- **Token Attribution Heatmap**: Shows which words influence safety classification
  - Red = triggers "unsafe" classification
  - Blue = triggers "safe" classification
  
- **Layer Activations**: Shows activation magnitude across transformer layers
  - Spike after layer 20 = safety-specific processing
  
- **Circuit Comparison**: Compares safe vs unsafe prompt processing
  - Higher divergence = model distinguishes safe from unsafe
  
- **Neuron Map** (if Advanced MI enabled): Identifies safety-specialized neurons
  - Larger circles = stronger safety signals
  
- **Circuit Diagram** (if Advanced MI enabled): Shows information flow

### 3. Open-Ended Adversarial Testing

**What it does**: Discovers vulnerabilities by generating adversarial prompts

#### Settings
- **Open-Ended Iterations** (default: 100): Number of mutation attempts
- **Retrain Interval** (default: 100): Retrain on adversarial examples every N iterations
- **Visualization Interval** (default: 50): Generate archive heatmap every N iterations

#### How it Works
1. Starts with seed harmful prompts
2. Mutates them to create variations
3. Tests if the judge is fooled
4. Maintains an archive of successful attacks
5. Can retrain the model on these attacks

#### Archive Dimensions
- **Risk Categories**: violence_hate, criminal_planning, self_harm, fraud_scams, etc.
- **Evasion Techniques**: direct_request, role_play, hypotheticals, technical_jargon, etc.

### 4. Prompt Testing

Test individual prompts against the trained model:
- Shows prediction (SAFE/UNSAFE) with confidence
- Displays token attribution visualization
- Useful for understanding model decisions

## 🎯 Recommended Workflow

### Initial Training
1. Start with default settings
2. Train for 15 epochs
3. Watch visualizations evolve each epoch
4. Look for:
   - Increasing validation accuracy
   - Clear separation in circuit comparison
   - Consistent token attribution patterns

### Advanced Analysis
1. Enable "Advanced MI Analysis" for detailed neuron/circuit insights
2. This will slow training but provide:
   - Safety neuron identification
   - Circuit tracing
   - More detailed vulnerability insights

### Adversarial Testing
1. After training, run Open-Ended testing
2. Start with 100 iterations to probe vulnerabilities
3. Review the archive heatmap to see vulnerability patterns
4. Consider retraining on discovered attacks

## 📊 Understanding the Metrics

### Training Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Balance between precision and recall
- **AUC**: Area under ROC curve (robustness metric)
- **Loss**: Training objective (lower is better)

### Open-Ended Metrics
- **Successful Attacks**: Prompts that fooled the judge
- **Archive Coverage**: How much of the risk/evasion space is explored
- **Success Rate**: Percentage of mutations that succeed

## 🔧 Optimizing Performance

### For Faster Training
- Reduce batch size if GPU memory limited
- Increase gradient accumulation to maintain effective batch size
- Disable Advanced MI Analysis

### For Better Results
- Train for more epochs (20-30)
- Use larger effective batch size (increase gradient accumulation)
- Run Open-Ended testing to identify weaknesses
- Retrain on adversarial examples

### For Deeper Insights
- Enable Advanced MI Analysis
- Run Open-Ended for more iterations (500-1000)
- Analyze patterns in successful attacks

## 🚨 Troubleshooting

### Out of Memory
- Reduce batch size (try 2 or 1)
- Reduce max sequence length
- Disable Advanced MI Analysis

### Training Not Converging
- Lower learning rate (try 1e-4 or 5e-5)
- Increase epochs
- Check if data is properly prepared

### Open-Ended Not Finding Attacks
- Model might be robust!
- Try more iterations
- Check if model is overly conservative (classifies everything as unsafe)

## 📈 Interpreting Visualizations

### Token Attribution
- Look for consistent patterns in what triggers safety
- Check if model focuses on relevant keywords or context

### Layer Activations
- Sharp increase after frozen layers = model learned safety features
- Flat profile = model might not be learning

### Circuit Comparison
- Large divergence = model distinguishes well
- Early divergence = efficient safety detection
- No divergence = model treats all prompts similarly

### Neuron Maps
- Clustered neurons = specialized safety detection
- Scattered neurons = distributed processing
- Few active neurons = simple decision rules

## 🎓 Advanced Tips

1. **Curriculum Learning**: Start with easy examples, gradually increase difficulty
2. **Ensemble**: Train multiple models with different seeds
3. **Active Learning**: Use Open-Ended to find edge cases, label them, retrain
4. **Transfer Learning**: Start from a model trained on general safety data
5. **Multi-objective**: Balance safety detection with avoiding false positives

## 📝 Notes

- All visualizations are saved as HTML files in `outputs/checkpoints/training_visualizations/`
- Open-Ended results are saved in `outputs/open_ended/`
- Best model checkpoint is at `outputs/checkpoints/best_model.pt`
- Training logs show real-time progress in the dashboard

Happy training! 🚀 