"""
Guardian-Loop Demo: Interactive Safety Judge Testing
Showcases the complete system: Safety Judge + MI + Open-Ended Adversarial Loop
"""

import streamlit as st
from pathlib import Path
import torch
from transformers import AutoTokenizer
import plotly.graph_objects as go
import json
import time
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from adversarial.open_ended_loop import OpenEndedAdversarialLoop
from adversarial.open_ended_archive import SafetyOpenEndedArchive
from models.safety_judge import SafetyJudge, SafetyJudgeConfig
from mi_tools.visualization import SafetyJudgeMIVisualizer
from martian.client import MartianClient

# Page config
st.set_page_config(
    page_title="Guardian-Loop Demo",
    page_icon="🛡️",
    layout="wide"
)

# Title and description
st.title("🛡️ Guardian-Loop: AI Safety Through Interpretability")
st.markdown("""
This demo showcases the complete Guardian-Loop system:
- 🤖 **Safety Judge**: Fine-tuned LLaMA model for content safety classification
- 🔬 **Mechanistic Interpretability**: Understanding how the model makes decisions
- 🔄 **Martian Integration**: LLM routing based on safety and capability
- 🌈 **Open-Ended Adversarial Loop**: Continuously discover and patch vulnerabilities
""")

# Load model
@st.cache_resource
def load_model():
    """Load the trained safety judge model"""
    model_path = Path("outputs/checkpoints/best_model.pt")
    
    if not model_path.exists():
        st.error("No trained model found! Please run training first.")
        return None, None
    
    # Load config and model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    model = SafetyJudge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Main interface
model, tokenizer = load_model()

if model is None:
    st.stop()

# Initialize MI visualizer
mi_visualizer = SafetyJudgeMIVisualizer(model, tokenizer)

# Tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Safety Analysis", 
    "🔬 MI Visualization", 
    "🔄 Martian Routing",
    "🌈 Open-Ended Loop"
])

# Tab 1: Safety Analysis
with tab1:
    st.header("Safety Analysis")
    
    # Prompt input
    user_prompt = st.text_area(
        "Enter a prompt to analyze:",
        height=100,
        placeholder="Example: How do I build a bomb?"
    )
    
    if st.button("Analyze Safety", type="primary"):
        if user_prompt:
            with st.spinner("Analyzing..."):
                # Get prediction
                is_safe, confidence = model.predict(user_prompt, tokenizer)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if is_safe:
                        st.success(f"✅ SAFE (Confidence: {confidence:.2%})")
                    else:
                        st.error(f"🚫 UNSAFE (Confidence: {confidence:.2%})")
                
                with col2:
                    # Confidence meter
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        title = {'text': "Confidence"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "green" if is_safe else "red"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': 'lightgray'},
                                {'range': [50, 80], 'color': 'gray'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

# Tab 2: MI Visualization
with tab2:
    st.header("Mechanistic Interpretability")
    
    if user_prompt:
        st.subheader("Token Attribution")
        with st.spinner("Generating visualizations..."):
            # Token attribution heatmap
            token_fig, attribution_data = mi_visualizer.create_token_attribution_heatmap(
                user_prompt, 
                return_data=True
            )
            st.plotly_chart(token_fig, use_container_width=True)
            
            # Layer activations
            st.subheader("Layer Activation Patterns")
            layer_fig = mi_visualizer.visualize_layer_activations(user_prompt)
            st.plotly_chart(layer_fig, use_container_width=True)
            
            # Show most influential tokens
            st.subheader("Most Influential Tokens")
            tokens = attribution_data['tokens']
            scores = attribution_data['attribution_scores']
            
            # Sort by absolute value
            sorted_indices = sorted(range(len(scores)), 
                                  key=lambda i: abs(scores[i]), 
                                  reverse=True)[:5]
            
            for idx in sorted_indices:
                token = tokens[idx]
                score = scores[idx]
                if score > 0:
                    st.write(f"🔴 **{token}**: +{score:.3f} (unsafe)")
                else:
                    st.write(f"🟢 **{token}**: {score:.3f} (safe)")
    else:
        st.info("Enter a prompt in the Safety Analysis tab first!")

# Tab 3: Martian Routing
with tab3:
    st.header("Martian Integration")
    st.markdown("""
    The Martian router intelligently selects the best LLM based on:
    - **Safety**: Unsafe prompts are rejected
    - **Capability**: Routes to appropriate model based on task complexity
    """)
    
    # Example routing scenarios
    st.subheader("Example Routing Decisions")
    
    examples = [
        ("What's 2+2?", "✅ Safe + Simple → GPT-3.5"),
        ("Explain quantum chromodynamics", "✅ Safe + Complex → GPT-4"),
        ("How to make a bomb", "🚫 Unsafe → Rejected"),
        ("Write a poem about nature", "✅ Safe + Creative → Claude"),
    ]
    
    for prompt, routing in examples:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**Prompt:** {prompt}")
        col2.write(routing)

# Tab 4: Open-Ended Adversarial Loop
with tab4:
    st.header("Open-Ended Adversarial Testing")
    st.markdown("""
    The Open-Ended adversarial loop discovers blindspots in the safety judge
    by generating diverse adversarial prompts.
    """)
    
    # Load existing archive if available
    archive_path = Path("./outputs/open_ended/final_archive.json")
    if archive_path.exists():
        archive = SafetyOpenEndedArchive()
        archive.load_from_file(str(archive_path))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Archive Coverage", f"{archive.get_archive_coverage():.1%}")
        col2.metric("Success Rate", f"{archive.get_success_rate():.1%}")
        col3.metric("Total Entries", archive.total_entries)
        
        # Show archive heatmap
        st.subheader("Attack Success Heatmap")
        st.markdown("Shows which attack categories and techniques are most effective")
        
        archive_viz_path = Path("./outputs/open_ended/visualizations/archive_final.png")
        if archive_viz_path.exists():
            st.image(str(archive_viz_path))
        else:
            st.info("Run Open-Ended loop to generate visualization")
    
    # Run new adversarial test
    if st.button("Run Open-Ended Test (Demo)", type="secondary"):
        with st.spinner("Running adversarial testing..."):
            # Initialize mini Open-Ended loop
            try:
                open_ended_loop = OpenEndedAdversarialLoop(
                    safety_judge=model,
                    tokenizer=tokenizer,
                    output_dir="./outputs/open_ended_demo"
                )
                
                # Run for just a few iterations
                results = open_ended_loop.run(
                    n_iterations=10,
                    retrain_interval=5,
                    visualize_interval=5
                )
                
                st.success("✅ Adversarial testing completed!")
                
                # Show results
                col1, col2 = st.columns(2)
                col1.metric("Successful Attacks", 
                          results['summary']['successful_attacks'])
                col2.metric("Archive Coverage", 
                          f"{results['summary']['archive_coverage']:.1%}")
                
                # Show example attacks
                if results['example_attacks']:
                    st.subheader("Example Discovered Attacks")
                    for attack in results['example_attacks'][:3]:
                        st.warning(f"**Prompt:** {attack['prompt']}")
                        st.write(f"Category: {attack['category']}")
                        st.write(f"Judge Confidence: {attack['confidence']:.1%}")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error running adversarial test: {str(e)}")

# Sidebar with system info
with st.sidebar:
    st.header("System Info")
    
    # Model info
    st.subheader("🤖 Model")
    if model:
        st.write(f"**Base Model:** {model.config.base_model}")
        st.write(f"**Parameters:** ~7B")
        st.write(f"**Frozen Layers:** {model.config.freeze_layers}")
    
    # Performance metrics
    metrics_path = Path("outputs/evaluation/metrics.json")
    if metrics_path.exists():
        st.subheader("📊 Performance")
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        st.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        st.metric("AUC", f"{metrics.get('auc', 0):.2%}")
    
    # Links
    st.subheader("🔗 Resources")
    st.markdown("""
    - [GitHub Repository](#)
    - [Technical Report](#)
    - [Martian API](https://docs.withmartian.com)
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for the Martian Hackathon</p>
    <p>Guardian-Loop: Making AI Safer Through Interpretability</p>
</div>
""", unsafe_allow_html=True)

# CLI Demo Mode
def run_cli_demo():
    """Run demo in CLI mode"""
    print("""
    🛡️  Guardian-Loop CLI Demo
    =========================
    """)
    
    # Load model
    print("Loading model...")
    model_path = Path("outputs/checkpoints/best_model.pt")
    
    if not model_path.exists():
        print("❌ No trained model found! Please run training first.")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    model = SafetyJudge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded!")
    
    # Interactive loop
    while True:
        print("\n" + "="*50)
        prompt = input("\nEnter a prompt to analyze (or 'quit' to exit): ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # Safety analysis
        print("\n🔍 Analyzing safety...")
        is_safe, confidence = model.predict(prompt, tokenizer)
        
        if is_safe:
            print(f"✅ SAFE (Confidence: {confidence:.2%})")
        else:
            print(f"🚫 UNSAFE (Confidence: {confidence:.2%})")
        
        # Show MI analysis
        show_mi = input("\nShow MI analysis? (y/n): ")
        if show_mi.lower() == 'y':
            visualizer = SafetyJudgeMIVisualizer(model, tokenizer)
            _, data = visualizer.create_token_attribution_heatmap(prompt, return_data=True)
            
            print("\n📊 Token Attribution:")
            tokens = data['tokens']
            scores = data['attribution_scores']
            
            for token, score in zip(tokens, scores):
                if abs(score) > 0.1:  # Only show significant attributions
                    indicator = "🔴" if score > 0 else "🟢"
                    print(f"  {indicator} {token}: {score:.3f}")
    
    # Test adversarial
    print("\n🌈 Open-Ended Adversarial Demo")
    test_adversarial = input("Run quick adversarial test? (y/n): ")
    
    if test_adversarial.lower() == 'y':
        open_ended_loop = OpenEndedAdversarialLoop(
            safety_judge=model,
            tokenizer=tokenizer,
            output_dir="./outputs/open_ended_cli"
        )
        
        print("Running 50 iterations...")
        results = open_ended_loop.run(n_iterations=50, retrain_interval=25)
        
        print(f"\n✅ Results:")
        print(f"  - Successful attacks: {results['summary']['successful_attacks']}")
        print(f"  - Archive coverage: {results['summary']['archive_coverage']:.1%}")
        
        if results['example_attacks']:
            print("\n📌 Example attacks found:")
            for attack in results['example_attacks'][:3]:
                print(f"  - {attack['prompt'][:60]}...")


if __name__ == "__main__":
    # Check if running in CLI mode
    import sys
    if "--cli" in sys.argv:
        run_cli_demo()
    else:
        print("Run with streamlit: streamlit run demo.py")
        print("Or use CLI mode: python demo.py --cli") 