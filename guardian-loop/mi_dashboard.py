"""
Streamlit Dashboard for Mechanistically Interpretable Micro Judges
Sequential interface with training controls and large visualizations
"""

import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import json
import re
from typing import Dict, List, Optional
import subprocess
import threading
import queue
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="MI Micro Judges",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.title("🛡️ Guardian-Loop Dashboard")
st.subheader("Safety Judge Training & Mechanistic Interpretability Analysis")

# Introduction section
with st.expander("📖 About Guardian-Loop", expanded=True):
    st.markdown("""
    **Guardian-Loop** is a comprehensive AI safety system with three main components:
    
    ### 1. 🤖 Safety Judge Model
    - Fine-tuned LLaMA model that classifies prompts as safe or unsafe
    - Uses prompt-based classification with True/False outputs
    - Trained on multiple safety datasets (WildGuard, BeaverTails, ToxicChat, etc.)
    
    ### 2. 🔬 Mechanistic Interpretability (MI)
    - **Token Attribution**: Shows which words trigger safety concerns
    - **Layer Activations**: Visualizes how different model layers respond
    - **Circuit Analysis**: Compares processing pathways for safe vs unsafe content
    - **Neuron Mapping**: Identifies specific neurons specialized for safety detection
    - **Circuit Diagrams**: Maps information flow through safety detection circuits
    
    ### 3. Open-Ended Adversarial Testing
    - Open-ended system for discovering model vulnerabilities
    - Generates adversarial prompts through mutations
    - Maintains an archive organized by:
      - **Risk Categories**: violence, criminal planning, self-harm, fraud, etc.
      - **Evasion Techniques**: role-play, hypotheticals, technical jargon, etc.
    - Can retrain the model on discovered vulnerabilities
    
    ### 🔄 The Loop
    1. **Train** the safety judge on curated datasets
    2. **Analyze** its behavior with MI tools
    3. **Attack** it with Open-Ended to find vulnerabilities
    4. **Improve** by retraining on discovered weaknesses
    5. **Repeat** for continuous improvement
    
    This dashboard provides controls for all components and real-time visualization of the training process.
    """)

# Helper functions
def parse_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch number from filename"""
    match = re.search(r'epoch_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def load_visualizations_by_epoch(viz_dir: Path) -> Dict[int, Dict[str, Path]]:
    """Load visualizations organized by epoch"""
    epochs = {}
    
    if not viz_dir.exists():
        return epochs
    
    for file in viz_dir.glob("*.html"):
        epoch = parse_epoch_from_filename(file.name)
        if epoch is not None:
            if epoch not in epochs:
                epochs[epoch] = {}
            
            # Categorize by type
            if 'tokens' in file.name:
                viz_type = 'token_attribution'
            elif 'layers' in file.name:
                viz_type = 'layer_activation'
            elif 'circuit_comparison' in file.name:
                viz_type = 'circuit_comparison'
            elif 'neuron_map' in file.name:
                viz_type = 'neuron_map'
            elif 'circuits' in file.name and 'comparison' not in file.name:
                viz_type = 'circuit_diagram'
            else:
                continue
            
            epochs[epoch][viz_type] = file
    
    return epochs

def display_visualization_iframe(file_path: Path, height: int = 800):
    """Display visualization using iframe with proper height"""
    if file_path.exists():
        with open(file_path, 'r') as f:
            html_content = f.read()
        
        import streamlit.components.v1 as components
        components.html(html_content, height=height, scrolling=True)
    else:
        st.error(f"Visualization file not found: {file_path}")

def run_training_pipeline(params: Dict):
    """Run the training pipeline with given parameters"""
    # Build command for train_safety_judge.py directly
    cmd = ["python", "guardian-loop/src/train_safety_judge.py"]
    
    # Add all parameters
    cmd.extend(["--num_epochs", str(params['num_epochs'])])
    cmd.extend(["--batch_size", str(params['batch_size'])])
    cmd.extend(["--learning_rate", str(params['learning_rate'])])
    cmd.extend(["--gradient_accumulation_steps", str(params['gradient_accumulation'])])
    cmd.extend(["--weight_decay", str(params['weight_decay'])])
    cmd.extend(["--freeze_layers", str(params['freeze_layers'])])
    cmd.extend(["--max_length", str(params['max_length'])])
    
    # Always enable visualization for the dashboard with interval 1
    cmd.append("--visualize_during_training")
    cmd.extend(["--visualization_interval", "1"])  # Every epoch
    
    if params['use_wandb']:
        cmd.append("--use_wandb")
    if params['run_advanced_mi']:
        cmd.append("--run_advanced_mi")
    
    # Add output directory
    cmd.extend(["--output_dir", "./outputs/checkpoints"])
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Main interface
st.divider()

# Check data preparation status
data_prepared = Path("data/prepared/train.json").exists() and Path("data/prepared/val.json").exists()

if not data_prepared:
    st.warning("""
    ⚠️ **Data not prepared!** Please run data preparation first:
    ```bash
    python guardian-loop/src/data/prepare_safety_data.py
    ```
    This will download and prepare safety datasets for training.
    """)
else:
    # Show data statistics
    try:
        with open("data/prepared/train.json", 'r') as f:
            train_data = json.load(f)
        with open("data/prepared/val.json", 'r') as f:
            val_data = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", len(train_data))
        col2.metric("Validation Samples", len(val_data))
        col3.metric("Total Samples", len(train_data) + len(val_data))
    except:
        pass

# Check if visualizations exist
viz_dir = Path("outputs/checkpoints/training_visualizations")
epochs_data = load_visualizations_by_epoch(viz_dir)

# SECTION 1: Training Parameters & Control
st.header("⚙️ Training Configuration")

# Store training state in session state
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'training_log' not in st.session_state:
    st.session_state.training_log = []

# Training parameters
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Data & Training")
    num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=50, value=15, 
                                help="Number of training epochs (default: 15)")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=4,
                                help="Training batch size (default: 4)")
    gradient_accumulation = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=16, value=4,
                                          help="Accumulate gradients over N steps (effective batch = batch_size × this)")

with col2:
    st.subheader("🎯 Optimization")
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-4, format="%.6f",
                                   help="Learning rate for training (default: 2e-4)")
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, format="%.3f",
                                  help="L2 regularization strength (default: 0.01)")
    freeze_layers = st.number_input("Freeze Layers", min_value=0, max_value=32, value=20,
                                   help="Number of transformer layers to freeze (default: 20)")

with col3:
    st.subheader("🔧 Advanced Options")
    max_length = st.number_input("Max Sequence Length", min_value=64, max_value=512, value=256,
                                help="Maximum token length for inputs (default: 256)")
    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=10, value=5,
                              help="Stop training if no improvement for N epochs")
    use_wandb = st.checkbox("Use Weights & Biases", value=False, 
                           help="Enable W&B logging (requires wandb login)")
    run_advanced_mi = st.checkbox("Advanced MI Analysis", value=False, 
                                 help="Run neuron & circuit analysis during training (slower but more detailed)")

st.info("📊 **Note:** MI visualizations are automatically generated every epoch during training to track model evolution!")

# Training control buttons
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("🚀 Start Training", type="primary", 
                 disabled=(st.session_state.training_process is not None or not data_prepared)):
        if not data_prepared:
            st.error("Cannot start training - data not prepared!")
        else:
            params = {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'gradient_accumulation': gradient_accumulation,
                'weight_decay': weight_decay,
                'freeze_layers': freeze_layers,
                'max_length': max_length,
                'patience': patience,
                'use_wandb': use_wandb,
                'run_advanced_mi': run_advanced_mi
            }
            
            st.session_state.training_process = run_training_pipeline(params)
            st.session_state.training_log = ["Training started..."]
            st.rerun()

with col2:
    if st.button("🛑 Stop Training", disabled=st.session_state.training_process is None):
        if st.session_state.training_process:
            st.session_state.training_process.terminate()
            st.session_state.training_process = None
            st.session_state.training_log.append("Training stopped by user.")
        st.rerun()

# Training status and log
if st.session_state.training_process is not None:
    st.info("🔄 Training in progress... Visualizations will be generated for EVERY epoch!")
    st.info("📊 You can refresh the page to see new visualizations as they are generated.")
    
    # Check if process is still running
    if st.session_state.training_process.poll() is not None:
        st.session_state.training_process = None
        st.success("✅ Training completed!")
        st.balloons()
        time.sleep(1)
        st.rerun()
    else:
        # Show last few lines of output
        try:
            line = st.session_state.training_process.stdout.readline()
            if line:
                st.session_state.training_log.append(line.strip())
                if len(st.session_state.training_log) > 20:
                    st.session_state.training_log = st.session_state.training_log[-20:]
        except:
            pass

# Show training log
if st.session_state.training_log:
    with st.expander("📜 Training Log", expanded=st.session_state.training_process is not None):
        st.code("\n".join(st.session_state.training_log[-10:]), language="text")

st.divider()

# SECTION 2: Visualizations (only show if data exists)
if epochs_data:
    st.header("📊 Training Visualizations")
    
    # Timeline with metrics
    sorted_epochs = sorted(epochs_data.keys())
    
    # Epoch selector - static, doesn't auto-update
    if 'selected_epoch' not in st.session_state:
        st.session_state.selected_epoch = sorted_epochs[-1]
    
    # Manual epoch selection
    st.subheader("📅 Training Timeline")
    
    # Display available epochs info
    st.info(f"Available epochs: {min(sorted_epochs)} - {max(sorted_epochs)} (Total: {len(sorted_epochs)} epochs)")
    
    # Epoch slider
    selected_epoch = st.select_slider(
        "Select Epoch to Visualize",
        options=sorted_epochs,
        value=st.session_state.selected_epoch,
        format_func=lambda x: f"Epoch {x}",
        key="epoch_slider"
    )
    st.session_state.selected_epoch = selected_epoch
    
    # Load and display metrics for selected epoch
    metrics_file = viz_dir.parent / f"epoch_{selected_epoch}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Display train/val metrics under timeline
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Train Accuracy", f"{metrics.get('train_accuracy', metrics.get('accuracy', 0)):.2%}")
        col2.metric("Val Accuracy", f"{metrics.get('val_accuracy', metrics.get('accuracy', 0)):.2%}")
        col3.metric("Train Loss", f"{metrics.get('train_loss', metrics.get('loss', 0)):.3f}")
        col4.metric("Val Loss", f"{metrics.get('val_loss', metrics.get('loss', 0)):.3f}")
        col5.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        col6.metric("AUC", f"{metrics.get('auc', 0):.2%}")
    
    # Refresh button
    if st.button("🔄 Refresh Visualizations"):
        st.rerun()
    
    st.divider()
    
    # VISUALIZATIONS - FULL WIDTH AND TALL
    epoch_visualizations = epochs_data.get(selected_epoch, {})
    
    # Token Attribution
    st.subheader("🔤 Token Attribution Heatmap")
    with st.expander("📚 Understanding Token Attribution", expanded=False):
        st.markdown("""
        **What does this visualization show?**
        
        This heatmap reveals which words (tokens) in the input prompt most influence the model's safety classification decision.
        
        **How to interpret:**
        - 🔴 **Red tokens** → Words that push the model toward classifying as "unsafe"
        - 🔵 **Blue tokens** → Words that push the model toward classifying as "safe"
        - **Color intensity** → Stronger colors mean higher influence on the decision
        - **White/neutral** → Words that don't significantly affect the classification
        
        **Key insights:**
        - Look for patterns in what words trigger safety concerns
        - Notice if the model focuses on context or specific keywords
        - Check if seemingly innocent words unexpectedly influence the decision
        
        **Example:** In "How to make a bomb", the word "bomb" would likely be bright red, while "How" might be neutral.
        """)
    
    if 'token_attribution' in epoch_visualizations:
        display_visualization_iframe(epoch_visualizations['token_attribution'], height=600)
    else:
        st.warning("No token attribution visualization for this epoch")
    
    st.divider()
    
    # Layer Activations
    st.subheader("📊 Layer Activation Patterns")
    with st.expander("📚 Understanding Layer Activations", expanded=False):
        st.markdown("""
        **What does this visualization show?**
        
        This graph displays how strongly different layers of the neural network activate when processing the input.
        
        **How to interpret:**
        - **X-axis** → Transformer layers (0 = input, higher = deeper processing)
        - **Y-axis** → Activation magnitude (higher = more neural activity)
        - **Red dashed line** → Boundary between frozen (pre-trained) and fine-tuned layers
        
        **Key insights:**
        - **Early layers** (0-10): Basic language understanding
        - **Middle layers** (10-20): Semantic comprehension
        - **Late layers** (20+): Safety-specific reasoning
        - **Spikes** indicate layers doing heavy processing for safety detection
        
        **What to look for:**
        - Sharp increases after the frozen/unfrozen boundary show safety-specific learning
        - Consistent patterns across epochs indicate stable safety detection mechanisms
        """)
    
    if 'layer_activation' in epoch_visualizations:
        display_visualization_iframe(epoch_visualizations['layer_activation'], height=700)
    else:
        st.warning("No layer activation visualization for this epoch")
    
    st.divider()
    
    # Circuit Comparison
    st.subheader("🔄 Safe vs Unsafe Circuit Analysis")
    with st.expander("📚 Understanding Circuit Comparison", expanded=False):
        st.markdown("""
        **What does this visualization show?**
        
        This analysis compares how the model's internal "circuits" (computational pathways) differ when processing safe vs unsafe content.
        
        **Two key graphs:**
        
        1. **Embedding Divergence (top)**
           - Shows how different the model's internal representations become
           - Higher divergence = model strongly distinguishes safe from unsafe
           - Peak indicates the "decision point" layer
        
        2. **Cosine Similarity (bottom)**
           - Measures how similar the processing pathways remain
           - Lower similarity = more specialized safety detection
           - Sharp drops indicate safety-specific processing kicks in
        
        **Key insights:**
        - **Critical layers**: Where divergence peaks is where safety detection happens
        - **Early convergence**: Model quickly identifies safety-relevant features
        - **Sustained divergence**: Strong, consistent safety classification
        
        **Ideal pattern**: Early divergence that remains high, with low similarity in later layers.
        """)
    
    if 'circuit_comparison' in epoch_visualizations:
        display_visualization_iframe(epoch_visualizations['circuit_comparison'], height=900)
    else:
        st.warning("No circuit comparison visualization for this epoch")
    
    st.divider()
    
    # Neuron Maps
    st.subheader("🗺️ Safety Neuron Activation Map")
    with st.expander("📚 Understanding Neuron Maps", expanded=False):
        st.markdown("""
        **What does this visualization show?**
        
        This scatter plot identifies individual neurons that specialize in detecting unsafe content.
        
        **How to interpret:**
        - **X-axis** → Layer number (depth in the network)
        - **Y-axis** → Neuron index (which neuron in that layer)
        - **Circle size** → How strongly the neuron responds to safety signals
        - **Color intensity** → Strength of the safety detection signal
        
        **Key insights:**
        - **Clusters** indicate groups of neurons working together for safety
        - **Large circles** are "safety specialist" neurons
        - **Distribution** shows if safety detection is localized or distributed
        
        **Interactive features:**
        - Hover over circles to see:
          - Exact neuron location
          - Activation strength
          - Top words that trigger this neuron
        
        **What to look for:**
        - Neurons in layers 20-26 often specialize in safety
        - Consistent patterns across training indicate robust safety mechanisms
        """)
    
    if 'neuron_map' in epoch_visualizations:
        display_visualization_iframe(epoch_visualizations['neuron_map'], height=800)
    else:
        st.warning("No neuron map visualization for this epoch")
    
    st.divider()
    
    # Circuit Diagrams
    st.subheader("🔌 Safety Detection Circuit Diagram")
    with st.expander("📚 Understanding Circuit Diagrams", expanded=False):
        st.markdown("""
        **What does this visualization show?**
        
        This network diagram maps the computational "circuits" - connected pathways of neurons that work together to detect unsafe content.
        
        **How to interpret:**
        - **Nodes** → Individual neurons (labeled as L{layer}_N{neuron})
        - **Edges** → Information flow between neurons
        - **Colors** → Different circuits (each color is a separate detection pathway)
        - **Layout** → Neurons cluster based on their connections
        
        **Key insights:**
        - **Multiple circuits** provide redundancy in safety detection
        - **Circuit depth** shows how many processing steps for safety decisions
        - **Convergence points** indicate critical decision neurons
        
        **What to look for:**
        - Dense connections suggest robust safety detection
        - Parallel circuits mean multiple ways to catch unsafe content
        - Isolated neurons might indicate specialized detection for specific threats
        
        **Example:** A circuit might flow: L20_N5 → L21_N12 → L22_N3, forming a pathway that detects violent content.
        """)
    
    if 'circuit_diagram' in epoch_visualizations:
        display_visualization_iframe(epoch_visualizations['circuit_diagram'], height=800)
    else:
        st.warning("No circuit diagram visualization for this epoch")

else:
    # No visualizations yet
    st.info("📊 No visualizations available yet. Configure and start training above to generate visualizations for EVERY epoch!")

st.divider()

# SECTION 3: Open-Ended Adversarial Testing
st.header("🌈 Open-Ended Adversarial Testing")

with st.expander("ℹ️ What is Open-Ended?", expanded=False):
    st.markdown("""
    **Open-Ended** is an open-ended adversarial prompt generation system that:
    
    - 🎯 **Discovers vulnerabilities** by generating adversarial prompts that fool the safety judge
    - 🧬 **Uses mutations** to explore the space of possible harmful prompts
    - 📊 **Maintains an archive** of successful attacks organized by risk category and evasion technique
    - 🔄 **Can retrain the model** on discovered vulnerabilities to improve robustness
    - 📈 **Tracks evolution** of attack patterns across generations
    
    The system explores two dimensions:
    1. **Risk Categories**: violence/hate, criminal planning, self-harm, fraud/scams, etc.
    2. **Evasion Techniques**: role-play, hypotheticals, technical jargon, authority appeals, etc.
    """)

# Open-Ended configuration
col1, col2 = st.columns(2)

with col1:
    open_ended_iterations = st.number_input("Open-Ended Iterations", min_value=10, max_value=10000, value=100,
                                       help="Number of iterations to run")
    retrain_interval = st.number_input("Retrain Interval", min_value=10, max_value=1000, value=100,
                                     help="Retrain model on adversarial examples every N iterations")

with col2:
    visualize_interval = st.number_input("Visualization Interval", min_value=10, max_value=100, value=50,
                                       help="Generate Open-Ended visualizations every N iterations")
    output_dir = st.text_input("Output Directory", value="outputs/open_ended",
                             help="Where to save Open-Ended results")

# Check if model exists for Open-Ended testing
model_exists = Path("outputs/checkpoints/best_model.pt").exists()

if st.button("🌈 Run Open-Ended Testing", type="secondary", disabled=not model_exists):
    if not model_exists:
        st.error("❌ No trained model found! Please train a model first.")
    else:
        with st.spinner("Running Open-Ended adversarial testing..."):
            result = subprocess.run([
                "python", "guardian-loop/src/adversarial/open_ended_loop.py",
                "--iterations", str(open_ended_iterations),
                "--output_dir", output_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("✅ Open-Ended testing completed!")
                
                # Try to load and display results
                try:
                    stats_file = Path(output_dir) / "final_statistics.json"
                    if stats_file.exists():
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Successful Attacks", stats.get('successful_attacks', 0))
                        col2.metric("Archive Coverage", f"{stats.get('final_coverage', 0):.1%}")
                        col3.metric("Success Rate", f"{stats.get('final_success_rate', 0):.1%}")
                    
                    # Show archive visualization if available
                    archive_viz = Path(output_dir) / "visualizations" / f"archive_iter_{open_ended_iterations}.png"
                    if archive_viz.exists():
                        st.image(str(archive_viz), caption="Open-Ended Archive Heatmap")
                    
                except Exception as e:
                    st.warning(f"Could not load results: {e}")
                    
            else:
                st.error(f"❌ Open-Ended testing failed: {result.stderr}")

# Display existing Open-Ended results
open_ended_results_dir = Path("outputs/open_ended")
if open_ended_results_dir.exists() and any(open_ended_results_dir.iterdir()):
    st.subheader("📁 Previous Open-Ended Results")
    
    # Find all Open-Ended result directories
    result_dirs = [d for d in open_ended_results_dir.iterdir() if d.is_dir()]
    
    if result_dirs:
        selected_result = st.selectbox(
            "Select Result to View",
            options=result_dirs,
            format_func=lambda x: x.name
        )
        
        # Load and display selected result
        if selected_result:
            stats_file = selected_result / "final_statistics.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                st.json(stats)

st.divider()

# SECTION 4: Prompt Testing
st.header("🔍 Test Model")

test_prompt = st.text_area(
    "Enter a prompt to analyze:",
    placeholder="Example: How to create a computer virus",
    height=100
)

if st.button("🔍 Analyze Prompt", type="primary", disabled=not Path("outputs/checkpoints/best_model.pt").exists()):
    if test_prompt:
        with st.spinner("Analyzing..."):
            # Escape quotes in the prompt
            escaped_prompt = test_prompt.replace('"', '\\"').replace("'", "\\'")
            
            # Create temporary script to run analysis
            script_content = f"""
import sys
sys.path.append('guardian-loop/src')
from mi_tools.visualization import SafetyJudgeMIVisualizer
from models.safety_judge import SafetyJudge, SafetyJudgeConfig
from transformers import AutoTokenizer
import torch
from pathlib import Path

checkpoint = torch.load('outputs/checkpoints/best_model.pt', map_location='cpu')
config = checkpoint['model_config']
model = SafetyJudge(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

visualizer = SafetyJudgeMIVisualizer(model, tokenizer)
test_prompt = '''{escaped_prompt}'''
fig, data = visualizer.create_token_attribution_heatmap(test_prompt, return_data=True)

print(f"RESULT:{{data['prediction']}}:{{data['confidence']}}")
fig.write_html('temp_analysis.html')
"""
            
            with open('temp_analyze.py', 'w') as f:
                f.write(script_content)
            
            result = subprocess.run(
                ['python', 'temp_analyze.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and "RESULT:" in result.stdout:
                # Parse result
                result_line = [l for l in result.stdout.split('\n') if 'RESULT:' in l][0]
                _, pred, conf = result_line.split(':')
                
                col1, col2 = st.columns(2)
                with col1:
                    if pred == "0":
                        st.error(f"🚫 UNSAFE (Confidence: {float(conf):.1%})")
                    else:
                        st.success(f"✅ SAFE (Confidence: {float(conf):.1%})")
                
                # Show visualization
                if Path('temp_analysis.html').exists():
                    st.subheader("Token Attribution Analysis")
                    display_visualization_iframe(Path('temp_analysis.html'), height=600)
            else:
                st.error(f"Analysis failed. Error: {result.stderr}")
                if result.stdout:
                    st.text("Output:")
                    st.code(result.stdout)
            
            # Cleanup
            Path('temp_analyze.py').unlink(missing_ok=True)
            Path('temp_analysis.html').unlink(missing_ok=True)

# Footer
st.divider()
st.caption("Guardian-Loop Dashboard - AI Safety through Mechanistic Interpretability & Adversarial Testing")

if __name__ == "__main__":
    # Auto-refresh every 5 seconds if training is in progress
    if st.session_state.get('training_process') is not None:
        time.sleep(5)
        st.rerun() 