"""
Mechanistically Interpretable Micro Judges Dashboard (Gradio Version)
Professional interface for AI safety model training and analysis
"""

import gradio as gr
import subprocess
import json
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import time
import threading
from typing import Dict, List, Optional, Tuple
import base64
import re


class MicroJudgesDashboard:
    def __init__(self):
        self.process = None
        self.output_log = []
        self.viz_dir = Path("outputs/checkpoints/training_visualizations")
        
    def run_pipeline(self, batch_size, learning_rate, num_epochs, grad_accum, 
                    freeze_layers, max_length, weight_decay, skip_data_prep, skip_training,
                    visualize_training, skip_open_ended, use_wandb, run_advanced_mi):
        """Run the training pipeline with specified parameters"""
        
        # Build command
        cmd = [
            "python", "src/train_safety_judge.py",
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--num_epochs", str(num_epochs),
            "--gradient_accumulation_steps", str(grad_accum),
            "--freeze_layers", str(freeze_layers),
            "--max_length", str(max_length),
            "--weight_decay", str(weight_decay)
        ]
        
        if visualize_training:
            cmd.extend(["--visualize_during_training", "--visualization_interval", "3"])
        
        if use_wandb:
            cmd.append("--use_wandb")
            
        if run_advanced_mi:
            cmd.append("--run_advanced_mi")
        
        # Start process
        self.output_log = ["Starting pipeline with parameters:"]
        self.output_log.append(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")
        self.output_log.append(f"Effective batch size: {batch_size * grad_accum}")
        if run_advanced_mi:
            self.output_log.append("Advanced MI analysis enabled (analyzing 5 safe/unsafe prompts)")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_log.append(line.strip())
                    if len(self.output_log) > 100:  # Keep last 100 lines
                        self.output_log.pop(0)
            
            self.process.wait()
            self.output_log.append("Pipeline completed successfully")
            
        except Exception as e:
            self.output_log.append(f"Error: {str(e)}")
            
        return "\n".join(self.output_log[-50:])  # Return last 50 lines
    
    def stop_pipeline(self):
        """Stop the running pipeline"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            return "Pipeline stopped"
        return "No pipeline running"
    
    def load_and_evaluate(self):
        """Load existing model and run evaluation with visualizations"""
        import subprocess
        
        output_log = ["Loading model and running evaluation..."]
        
        # Run evaluation
        cmd = ["python", "src/evaluate_safety.py", "--visualize"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                output_log.append("Evaluation complete successfully")
                output_log.append("Check visualizations tab for results")
                output_log.extend(result.stdout.split('\n')[-10:])  # Last 10 lines
            else:
                output_log.append(f"Evaluation failed: {result.stderr}")
                
        except Exception as e:
            output_log.append(f"Error: {str(e)}")
            
        return "\n".join(output_log)
    
    def analyze_prompt(self, prompt: str, show_mi: bool):
        """Analyze a single prompt with the trained model"""
        import subprocess
        from pathlib import Path
        
        if not prompt:
            return "Please enter a prompt to analyze", ""
        
        # Check if model exists
        model_path = Path("outputs/checkpoints/best_model.pt")
        if not model_path.exists():
            return "No trained model found. Please train a model first.", ""
        
        # Run inference
        cmd = ["python", "src/inference.py", "--prompt", prompt, "--json"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                output = json.loads(result.stdout)
                
                # Format result
                classification = output['classification'].upper()
                confidence = output['confidence']
                
                if output['is_safe']:
                    result_text = f"CLASSIFICATION: {classification}\nConfidence: {confidence:.1%}"
                else:
                    result_text = f"CLASSIFICATION: {classification}\nConfidence: {confidence:.1%}"
                
                # Generate MI visualization if requested
                mi_html = ""
                if show_mi:
                    mi_cmd = ["python", "src/mi_tools/visualization.py", 
                             "--mode", "single", "--prompt", prompt]
                    mi_result = subprocess.run(mi_cmd, capture_output=True, text=True)
                    
                    if mi_result.returncode == 0:
                        # Load the generated HTML
                        viz_path = Path("outputs/mi_visualizations/single_prompt_analysis.html")
                        if viz_path.exists():
                            with open(viz_path, 'r') as f:
                                mi_html = f.read()
                
                return result_text, mi_html
            else:
                return f"Analysis failed: {result.stderr}", ""
                
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def get_visualizations(self, viz_type: str) -> List[Tuple[str, str]]:
        """Get list of available visualizations of a specific type"""
        if not self.viz_dir.exists():
            return []
        
        patterns = {
            "token": "*tokens*.html",
            "layer": "*layers*.html",
            "circuit": "*comparison*.html",
            "neuron": "*neuron_map*.html",
            "circuits": "*circuits*.html",
            "metrics": "*progress*.html"
        }
        
        pattern = patterns.get(viz_type, "*.html")
        files = list(self.viz_dir.glob(pattern))
        
        return [(f.name.replace('_', ' ').title(), str(f)) for f in sorted(files)]
    
    def load_visualization(self, file_path: str) -> str:
        """Load and return visualization HTML"""
        if not file_path or file_path == "None":
            return "<p style='text-align: center; color: #666; padding: 2rem;'>Select a visualization from the dropdown above</p>"
        
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"<p style='color: #BF616A;'>Error loading visualization: {str(e)}</p>"


def create_interface():
    """Create the Gradio interface"""
    dashboard = MicroJudgesDashboard()
    
    # Custom CSS for professional styling
    custom_css = """
    /* Professional color scheme - Nord theme */
    :root {
        --brand-primary: #2E3440;
        --brand-accent: #5E81AC;
        --brand-highlight: #88C0D0;
        --brand-success: #A3BE8C;
        --brand-danger: #BF616A;
        --text-primary: #2E3440;
        --text-secondary: #4C566A;
        --bg-primary: #ECEFF4;
        --bg-secondary: #E5E9F0;
    }
    
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: var(--bg-primary);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #2E3440 0%, #3B4252 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .main-header h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #88C0D0;
        margin: 0.5rem 0 0 0;
    }
    
    .main-header p {
        font-size: 1rem;
        color: #D8DEE9;
        margin: 1rem 0 0 0;
    }
    
    /* Button styling */
    .gr-button-primary {
        background-color: var(--brand-accent) !important;
        border-color: var(--brand-accent) !important;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    .gr-button-primary:hover {
        background-color: var(--brand-highlight) !important;
        border-color: var(--brand-highlight) !important;
    }
    
    .gr-button-secondary {
        background-color: transparent !important;
        border-color: var(--brand-accent) !important;
        color: var(--brand-accent) !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: var(--bg-secondary);
        border-left: 4px solid var(--brand-accent);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Tab styling */
    .gr-tab-button {
        font-weight: 500;
        text-transform: none;
    }
    
    .gr-tab-button-active {
        background-color: var(--brand-accent) !important;
        color: white !important;
    }
    
    /* Clean form styling */
    .gr-form {
        background-color: white;
        border: 1px solid #D8DEE9;
        border-radius: 4px;
    }
    
    .visualization-container {
        min-height: 600px;
        border: 1px solid #D8DEE9;
        border-radius: 4px;
        background: white;
        padding: 1rem;
    }
    """
    
    with gr.Blocks(
        title="MI Micro Judges",
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray"
        ),
        css=custom_css
    ) as demo:
        
        # Professional header
        gr.HTML("""
        <div class="main-header">
            <h1>MECHANISTICALLY INTERPRETABLE</h1>
            <h2>MICRO JUDGES</h2>
            <p>Advanced AI Safety Analysis Platform</p>
        </div>
        """)
        
        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                gr.Markdown("## Configuration Panel")
                
                with gr.Group():
                    gr.Markdown("### Training Parameters")
                    batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                    learning_rate = gr.Number(value=0.0002, label="Learning Rate", precision=4)
                    num_epochs = gr.Slider(1, 50, value=15, step=1, label="Epochs")
                    grad_accum = gr.Slider(1, 8, value=4, step=1, label="Gradient Accumulation")
                    freeze_layers = gr.Slider(0, 32, value=20, step=1, label="Freeze Layers")
                    max_length = gr.Slider(64, 512, value=256, step=64, label="Max Length")
                    weight_decay = gr.Number(value=0.01, label="Weight Decay", precision=3)
                    
                    effective_batch = gr.Markdown(f"**Effective Batch Size**: {8 * 4}")
                
                with gr.Group():
                    gr.Markdown("### Pipeline Options")
                    with gr.Row():
                        with gr.Column():
                            skip_data_prep = gr.Checkbox(label="Skip Data Preparation", value=False)
                            skip_training = gr.Checkbox(label="Skip Training", value=False)
                            visualize_training = gr.Checkbox(label="Generate MI Visualizations", value=True)
                        with gr.Column():
                            skip_open_ended = gr.Checkbox(label="Skip Open-Ended Testing", value=True)
                            use_wandb = gr.Checkbox(label="Use Weights & Biases", value=False)
                            run_advanced_mi = gr.Checkbox(
                                label="Run Advanced MI Analysis", 
                                value=False,
                                info="Provides detailed neuron/circuit analysis"
                            )
                
                with gr.Row():
                    run_btn = gr.Button("RUN PIPELINE", variant="primary")
                    stop_btn = gr.Button("STOP", variant="secondary")
                    eval_btn = gr.Button("LOAD & EVALUATE", variant="secondary")
                
                output_log = gr.Textbox(
                    label="Pipeline Output",
                    lines=10,
                    max_lines=20,
                    value="Ready to start...",
                    interactive=False
                )
                
                # Model Testing Section
                gr.Markdown("### Model Testing")
                
                with gr.Group():
                    test_prompt = gr.Textbox(
                        label="Enter a prompt to test:",
                        placeholder="Example: How to create a computer virus",
                        lines=3
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("ANALYZE", variant="primary")
                        show_mi = gr.Checkbox(label="Generate MI visualization", value=False)
                    
                    result_output = gr.Textbox(
                        label="Analysis Result",
                        lines=2,
                        interactive=False
                    )
                    
                    mi_viz_output = gr.HTML(
                        value="",
                        visible=False,
                        elem_classes="visualization-container"
                    )
            
            # Right column: Visualizations
            with gr.Column(scale=2):
                gr.Markdown("## Visualization Explorer")
                gr.Markdown("Explore mechanistic interpretability visualizations to understand model decisions")
                
                with gr.Tabs():
                    with gr.TabItem("Token Attribution"):
                        gr.HTML("""
                        <div class="info-box">
                            <strong>Token Attribution Heatmaps</strong><br>
                            Shows which tokens contribute most to safety classification.
                            Red indicates "unsafe" signals, blue indicates "safe" signals.
                        </div>
                        """)
                        token_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
                        token_viz = gr.HTML(
                            value="<p style='text-align: center; padding: 2rem;'>No visualizations available yet</p>",
                            elem_classes="visualization-container"
                        )
                    
                    with gr.TabItem("Layer Activations"):
                        gr.HTML("""
                        <div class="info-box">
                            <strong>Layer Activation Patterns</strong><br>
                            Visualizes activation magnitude across transformer layers.
                            Higher activations indicate layers more involved in safety detection.
                        </div>
                        """)
                        layer_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
                        layer_viz = gr.HTML(
                            value="<p style='text-align: center; padding: 2rem;'>No visualizations available yet</p>",
                            elem_classes="visualization-container"
                        )
                    
                    with gr.TabItem("Circuit Comparisons"):
                        gr.HTML("""
                        <div class="info-box">
                            <strong>Safe vs Unsafe Circuit Analysis</strong><br>
                            Compares processing patterns between safe and unsafe prompts.
                            Shows divergence and similarity metrics across layers.
                        </div>
                        """)
                        circuit_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
                        circuit_viz = gr.HTML(
                            value="<p style='text-align: center; padding: 2rem;'>No visualizations available yet</p>",
                            elem_classes="visualization-container"
                        )
                    
                    with gr.TabItem("Neuron Maps"):
                        gr.HTML("""
                        <div class="info-box">
                            <strong>Safety Neuron Activation Maps</strong><br>
                            Interactive plots showing neurons most active in unsafe content detection.
                            Larger points indicate stronger safety signals.
                        </div>
                        """)
                        neuron_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
                        neuron_viz = gr.HTML(
                            value="<p style='text-align: center; padding: 2rem;'>No visualizations available yet</p>",
                            elem_classes="visualization-container"
                        )
                    
                    with gr.TabItem("Training Metrics"):
                        gr.HTML("""
                        <div class="info-box">
                            <strong>Training Progress Metrics</strong><br>
                            Track model performance over training epochs.
                            Includes accuracy, F1 score, and loss curves.
                        </div>
                        """)
                        metrics_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
                        metrics_viz = gr.HTML(
                            value="<p style='text-align: center; padding: 2rem;'>No visualizations available yet</p>",
                            elem_classes="visualization-container"
                        )
        
        # Refresh button
        with gr.Row():
            gr.Markdown("")  # Spacer
            refresh_btn = gr.Button("REFRESH VISUALIZATIONS", variant="secondary")
            gr.Markdown("")  # Spacer
        
        # Event handlers
        def update_visualizations():
            """Update all visualization dropdowns"""
            return {
                token_dropdown: gr.update(choices=dashboard.get_visualizations("token")),
                layer_dropdown: gr.update(choices=dashboard.get_visualizations("layer")),
                circuit_dropdown: gr.update(choices=dashboard.get_visualizations("circuit")),
                neuron_dropdown: gr.update(choices=dashboard.get_visualizations("neuron")),
                metrics_dropdown: gr.update(choices=dashboard.get_visualizations("metrics"))
            }
        
        # Connect event handlers
        run_btn.click(
            fn=dashboard.run_pipeline,
            inputs=[batch_size, learning_rate, num_epochs, grad_accum,
                   freeze_layers, max_length, weight_decay, skip_data_prep, skip_training,
                   visualize_training, skip_open_ended, use_wandb, run_advanced_mi],
            outputs=output_log
        )
        
        stop_btn.click(
            fn=dashboard.stop_pipeline,
            outputs=output_log
        )
        
        eval_btn.click(
            fn=dashboard.load_and_evaluate,
            outputs=output_log
        )
        
        # Prompt analysis handlers
        def analyze_and_show_viz(prompt, show_mi):
            result, mi_html = dashboard.analyze_prompt(prompt, show_mi)
            return result, gr.update(value=mi_html, visible=show_mi)
        
        analyze_btn.click(
            fn=analyze_and_show_viz,
            inputs=[test_prompt, show_mi],
            outputs=[result_output, mi_viz_output]
        )
        
        # Show/hide MI visualization based on checkbox
        show_mi.change(
            fn=lambda x: gr.update(visible=x),
            inputs=show_mi,
            outputs=mi_viz_output
        )
        
        refresh_btn.click(
            fn=update_visualizations,
            outputs=[token_dropdown, layer_dropdown, circuit_dropdown, 
                    neuron_dropdown, metrics_dropdown]
        )
        
        # Visualization loaders
        token_dropdown.change(
            fn=dashboard.load_visualization,
            inputs=token_dropdown,
            outputs=token_viz
        )
        
        layer_dropdown.change(
            fn=dashboard.load_visualization,
            inputs=layer_dropdown,
            outputs=layer_viz
        )
        
        circuit_dropdown.change(
            fn=dashboard.load_visualization,
            inputs=circuit_dropdown,
            outputs=circuit_viz
        )
        
        neuron_dropdown.change(
            fn=dashboard.load_visualization,
            inputs=neuron_dropdown,
            outputs=neuron_viz
        )
        
        metrics_dropdown.change(
            fn=dashboard.load_visualization,
            inputs=metrics_dropdown,
            outputs=metrics_viz
        )
        
        # Update effective batch size dynamically
        def update_effective_batch(batch, accum):
            return f"**Effective Batch Size**: {batch * accum}"
        
        batch_size.change(update_effective_batch, [batch_size, grad_accum], effective_batch)
        grad_accum.change(update_effective_batch, [batch_size, grad_accum], effective_batch)
        
        # Auto-refresh visualizations on load
        demo.load(update_visualizations, outputs=[
            token_dropdown, layer_dropdown, circuit_dropdown,
            neuron_dropdown, metrics_dropdown
        ])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; color: #4C566A; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid #D8DEE9;">
            <p>Mechanistically Interpretable Micro Judges | Built for AI Safety Research</p>
            <p style="font-size: 0.875rem;">© 2024 Guardian-Loop Project</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    
    print("""
    ==================================================
    MECHANISTICALLY INTERPRETABLE MICRO JUDGES
    ==================================================
    
    Professional AI Safety Analysis Platform
    
    Access Options:
    - Local: http://localhost:7860
    - Remote SSH: ssh -L 7860:localhost:7860 user@remote
    - Public: Set share=True in demo.launch()
    
    ==================================================
    """)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True for public sharing via Gradio
        inbrowser=True
    ) 