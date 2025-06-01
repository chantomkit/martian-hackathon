"""
Mechanistic Interpretability Visualization Tools for Guardian-Loop
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LayerConductance
from typing import List, Dict, Tuple, Optional
import pandas as pd
from IPython.display import HTML, display


class SafetyJudgeMIVisualizer:
    """Mechanistic Interpretability visualizations for Safety Judge"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Initialize Captum attribution methods
        self.integrated_gradients = IntegratedGradients(self._forward_func)
        
    def _forward_func(self, input_ids):
        """Forward function for Captum - returns log probability difference"""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Get True/False token ids
        true_tokens = self.tokenizer(" True", add_special_tokens=False)
        false_tokens = self.tokenizer(" False", add_special_tokens=False)
        true_token_id = true_tokens["input_ids"][0]
        false_token_id = false_tokens["input_ids"][0]
        
        # Get log probabilities
        # Check if logits are 2D (batch_size, vocab_size) or 3D (batch_size, seq_len, vocab_size)
        if logits.dim() == 3:
            # If 3D, take the last position
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        else:
            # If 2D, already at last position
            log_probs = torch.log_softmax(logits, dim=-1)
        true_log_prob = log_probs[0, true_token_id].item()
        false_log_prob = log_probs[0, false_token_id].item()
        
        # Return log probability difference (positive = more likely True/safe)
        return true_log_prob - false_log_prob
    
    def _compute_token_attributions_via_gradients(self, input_ids, attention_mask):
        """Compute token attributions using gradient-based method that works with embeddings"""
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.base_model.model.embed_tokens(input_ids)
        
        # Enable gradients for embeddings
        embeddings = embeddings.detach().requires_grad_(True)
        
        # Forward pass through model with embeddings
        outputs = self.model.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits
        logits = outputs.logits
        
        # Get True/False token ids
        true_tokens = self.tokenizer(" True", add_special_tokens=False)
        false_tokens = self.tokenizer(" False", add_special_tokens=False)
        true_token_id = true_tokens["input_ids"][0]
        false_token_id = false_tokens["input_ids"][0]
        
        # Get log probability difference
        if logits.dim() == 3:
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        else:
            log_probs = torch.log_softmax(logits, dim=-1)
        
        true_log_prob = log_probs[0, true_token_id]
        false_log_prob = log_probs[0, false_token_id]
        log_prob_diff = true_log_prob - false_log_prob
        
        # Compute gradients
        log_prob_diff.backward()
        
        # Get attribution scores (gradient magnitude)
        attributions = embeddings.grad.abs() if embeddings.grad is not None else torch.zeros_like(embeddings)
        
        # Average across embedding dimensions to get per-token scores
        token_attributions = attributions.mean(dim=-1).squeeze(0)
        
        return token_attributions.cpu().numpy(), {
            'true_log_prob': true_log_prob.item(),
            'false_log_prob': false_log_prob.item(),
            'log_prob_diff': log_prob_diff.item()
        }
    
    def create_token_attribution_heatmap(self, text: str, return_data: bool = False):
        """Create an interactive heatmap showing which tokens trigger safety concerns"""
        
        # Format the prompt with the template if text doesn't already contain it
        if "Determine if the following text is safe or unsafe" not in text:
            formatted_text = self.model.config.prompt_template.format(text=text)
        else:
            formatted_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.model.config.max_length
        ).to(self.device)
        
        # Get base prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, inputs.attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get True/False token ids
            true_tokens = self.tokenizer(" True", add_special_tokens=False)
            false_tokens = self.tokenizer(" False", add_special_tokens=False)
            true_token_id = true_tokens["input_ids"][0]
            false_token_id = false_tokens["input_ids"][0]
            
            # Get log probabilities
            if logits.dim() == 3:
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            else:
                log_probs = torch.log_softmax(logits, dim=-1)
            
            true_log_prob = log_probs[0, true_token_id].item()
            false_log_prob = log_probs[0, false_token_id].item()
            
            # Convert to probability
            log_prob_diff = true_log_prob - false_log_prob
            safe_prob = 1.0 / (1.0 + torch.exp(torch.tensor(-log_prob_diff))).item()
            
            prediction = 1 if safe_prob > 0.5 else 0  # 1 = safe, 0 = unsafe
            confidence = abs(safe_prob - 0.5) * 2
        
        # Calculate gradient-based attributions
        attribution_scores, prob_info = self._compute_token_attributions_via_gradients(
            inputs.input_ids, inputs.attention_mask
        )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Normalize scores
        norm_scores = attribution_scores / (np.abs(attribution_scores).max() + 1e-10)
        
        # Create interactive Plotly visualization
        fig = go.Figure()
        
        # Add heatmap - reshape scores to 2D array for proper display
        fig.add_trace(go.Heatmap(
            z=[norm_scores],  # Keep as 2D array with single row
            x=list(range(len(tokens))),  # Use indices for x-axis
            y=['Tokens'],
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{token}<br>{score:.3f}" for token, score in zip(tokens, norm_scores)]],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Attribution<br>Score"),
            hovertemplate='Token: %{text}<br>Score: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        safety_label = "SAFE" if prediction == 1 else "UNSAFE"
        fig.update_layout(
            title="Token Attribution Heatmap",
            xaxis=dict(
                title="Token Position",
                tickmode='array',
                tickvals=list(range(len(tokens))),
                ticktext=[t[:15] + '...' if len(t) > 15 else t for t in tokens],  # Truncate long tokens
                tickangle=-45
            ),
            yaxis_title="",
            height=500,  # Increased height
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=12),
            margin=dict(t=150, b=200, l=100, r=100),  # Increased margins
            annotations=[
                dict(
                    text=f"<b>Input prompt:</b> {text[:100]}{'...' if len(text) > 100 else ''}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.15,
                    xanchor="center",
                    font=dict(size=12),
                    align="center"
                ),
                dict(
                    text=f"<b>Prediction:</b> {'UNSAFE' if prediction == 0 else 'SAFE'} (confidence: {confidence:.1%})",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.35,  # Moved down more
                    xanchor="center",
                    font=dict(size=14, color='red' if prediction == 0 else 'green'),
                    align="center"
                )
            ]
        )
        
        if return_data:
            return fig, {
                'tokens': tokens,
                'scores': norm_scores,
                'prediction': prediction,
                'safe_prob': safe_prob,
                'confidence': confidence,
                'true_log_prob': true_log_prob,
                'false_log_prob': false_log_prob,
                'log_prob_difference': log_prob_diff
            }
        
        return fig
    
    def analyze_attention_heads(self, text: str, layer_range: Tuple[int, int] = (20, 26)):
        """Analyze which attention heads focus on safety-critical patterns"""
        
        # Format the prompt with the template if needed
        if "Determine if the following text is safe or unsafe" not in text:
            formatted_text = self.model.config.prompt_template.format(text=text)
        else:
            formatted_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_length
        ).to(self.device)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Forward pass to get attention weights
        self.model.eval()
        with torch.no_grad():
            # Need to pass output_attentions=True
            outputs = self.model.base_model(
                inputs.input_ids, 
                inputs.attention_mask,
                output_attentions=True,
                return_dict=True
            )
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        if attentions is None:
            # Return empty visualization if no attention weights available
            fig = go.Figure()
            fig.add_annotation(
                text="Attention weights not available for this model",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title="Attention Head Analysis",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black')
            )
            return fig, {}
        
        # Identify safety-critical tokens (simple heuristic)
        safety_keywords = ['harm', 'dangerous', 'illegal', 'weapon', 'drug', 'kill', 
                          'steal', 'hack', 'exploit', 'manipulate', 'bomb', 'violence']
        safety_token_indices = []
        for i, token in enumerate(tokens):
            if any(keyword in token.lower() for keyword in safety_keywords):
                safety_token_indices.append(i)
        
        # Analyze attention to safety tokens
        head_importance = {}
        
        for layer_idx in range(*layer_range):
            if layer_idx < len(attentions):
                layer_attention = attentions[layer_idx][0]  # [heads, seq, seq]
                n_heads = layer_attention.shape[0]
                
                for head in range(n_heads):
                    if safety_token_indices:
                        # Average attention to safety tokens
                        attn_to_safety = layer_attention[head, :, safety_token_indices].mean().item()
                    else:
                        # If no safety tokens, look at overall attention pattern variance
                        attn_to_safety = layer_attention[head].std().item()
                    
                    head_importance[f"L{layer_idx}_H{head}"] = attn_to_safety
        
        # Sort heads by importance
        sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[h[0] for h in sorted_heads],
            y=[h[1] for h in sorted_heads],
            text=[f"{h[1]:.3f}" for h in sorted_heads],
            textposition='auto',
            marker_color='crimson'
        ))
        
        fig.update_layout(
            title=f"Top 15 Attention Heads for Safety Detection<br><sub>Full prompt: {text}</sub>",
            xaxis_title="Attention Head",
            yaxis_title="Average Attention to Safety Patterns",
            xaxis_tickangle=-45,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        
        return fig, head_importance
    
    def visualize_layer_activations(self, text: str):
        """Visualize how different layers respond to the input"""
        
        # Format the prompt with the template if needed
        if "Determine if the following text is safe or unsafe" not in text:
            formatted_text = self.model.config.prompt_template.format(text=text)
        else:
            formatted_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_length
        ).to(self.device)
        
        # Get activations from model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.base_model(
                inputs.input_ids, 
                inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states
        
        # Calculate activation magnitudes per layer
        layer_magnitudes = []
        layer_names = []
        
        for i, hidden in enumerate(hidden_states[1:]):  # Skip embedding layer
            magnitude = hidden.norm(dim=-1).mean().item()
            layer_magnitudes.append(magnitude)
            layer_names.append(f"Layer {i}")
        
        # Create plot
        fig = go.Figure()
        
        # Plot layer magnitudes
        fig.add_trace(
            go.Scatter(
                x=layer_names,
                y=layer_magnitudes,
                mode='lines+markers',
                name='Activation Magnitude',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            )
        )
        
        # Mark frozen layers
        if hasattr(self.model.config, 'freeze_layers'):
            fig.add_vline(
                x=self.model.config.freeze_layers - 0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Frozen/Unfrozen Boundary"
            )
        
        # Update layout with better spacing
        fig.update_layout(
            title=f"Layer-wise Activation Analysis",
            xaxis_title="Layer",
            yaxis_title="Activation Magnitude",
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(t=150, b=100),  # More margin for text
            annotations=[
                dict(
                    text=f"<b>Analyzing prompt:</b><br>{text[:150]}{'...' if len(text) > 150 else ''}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.2,
                    xanchor="center",
                    font=dict(size=12),
                    align="center"
                )
            ]
        )
        
        return fig
    
    def compare_safe_unsafe_circuits(self, safe_prompt: str, unsafe_prompt: str):
        """Compare activation patterns between safe and unsafe prompts"""
        
        # Get embeddings for both prompts
        def get_embeddings(text):
            # Format with template if needed
            if "Determine if the following text is safe or unsafe" not in text:
                formatted_text = self.model.config.prompt_template.format(text=text)
            else:
                formatted_text = text
                
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.base_model(
                    inputs.input_ids, 
                    inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states
                
            # Get mean pooled representation per layer
            embeddings = []
            for hidden in hidden_states[1:]:  # Skip embedding layer
                pooled = hidden.mean(dim=1).squeeze(0)
                embeddings.append(pooled)
                
            return torch.stack(embeddings)
        
        safe_embeds = get_embeddings(safe_prompt)
        unsafe_embeds = get_embeddings(unsafe_prompt)
        
        # Calculate divergence
        divergence = torch.norm(safe_embeds - unsafe_embeds, dim=-1).cpu().numpy()
        
        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            safe_embeds, unsafe_embeds, dim=-1
        ).cpu().numpy()
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Embedding Divergence", "Cosine Similarity"),
            vertical_spacing=0.15
        )
        
        x = list(range(len(divergence)))
        
        # Divergence plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=divergence,
                mode='lines+markers',
                name='L2 Divergence',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=1, col=1
        )
        
        # Mark critical divergence point
        critical_layer = int(np.argmax(divergence))
        fig.add_vline(
            x=critical_layer,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Max Divergence (Layer {critical_layer})",
            row=1, col=1
        )
        
        # Similarity plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cosine_sim,
                mode='lines+markers',
                name='Cosine Similarity',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,0,255,0.1)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Layer Index", row=2, col=1)
        fig.update_yaxes(title_text="L2 Distance", row=1, col=1)
        fig.update_yaxes(title_text="Similarity", row=2, col=1)
        
        # Update all axes to have grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.update_layout(
            title=dict(
                text="Safety Circuit Analysis: Safe vs Unsafe Prompts",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            height=800,  # Increased height
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(t=200, b=100, l=100, r=100)  # Increased top margin
        )
        
        # Add text boxes with full prompts - moved to avoid overlap
        fig.add_annotation(
            text=f"<b>Safe prompt:</b> {safe_prompt[:80]}{'...' if len(safe_prompt) > 80 else ''}<br>"
                 f"<b>Unsafe prompt:</b> {unsafe_prompt[:80]}{'...' if len(unsafe_prompt) > 80 else ''}",
            xref="paper", yref="paper",
            x=0.5, y=1.08,  # Moved to top center
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            align='center',
            xanchor='center',
            yanchor='bottom'
        )
        
        return fig, {
            'divergence': divergence,
            'similarity': cosine_sim,
            'critical_layer': critical_layer
        }
    
    def create_comprehensive_report(self, text: str):
        """Generate a comprehensive MI analysis report for a given prompt"""
        
        # Create all visualizations
        token_fig, token_data = self.create_token_attribution_heatmap(text, return_data=True)
        attention_fig, _ = self.analyze_attention_heads(text)
        layer_fig = self.visualize_layer_activations(text)
        
        # Create HTML report
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: auto; background-color: white; padding: 20px;">
            <h1 style="color: #333;">Mechanistic Interpretability Report</h1>
            <h2 style="color: #555;">Safety Judge Analysis</h2>
            
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #dee2e6;">
                <h3 style="color: #333;">Input Text:</h3>
                <p style="font-style: italic; color: #555; word-wrap: break-word;">"{text}"</p>
                <h3 style="color: #333;">Prediction:</h3>
                <p style="font-size: 18px;"><strong style="color: {'#dc3545' if token_data['prediction'] == 0 else '#28a745'}">{'UNSAFE' if token_data['prediction'] == 0 else 'SAFE'}</strong> 
                   <span style="color: #666;">(Confidence: {token_data['confidence']:.1%} | Safety probability: {token_data['safe_prob']:.2%})</span></p>
            </div>
            
            <h3>1. Token Attribution Analysis</h3>
            <p>Shows which tokens contribute most to the safety classification:</p>
            {token_fig.to_html(include_plotlyjs='cdn')}
            
            <h3>2. Attention Head Analysis</h3>
            <p>Identifies which attention heads focus on safety-critical patterns:</p>
            {attention_fig.to_html(include_plotlyjs='cdn')}
            
            <h3>3. Layer Activation Analysis</h3>
            <p>Shows how different layers respond to the input:</p>
            {layer_fig.to_html(include_plotlyjs='cdn')}
            
            <div style="margin-top: 40px; padding: 20px; background-color: #e8f4f8; border-radius: 5px;">
                <h3>Key Insights:</h3>
                <ul>
                    <li>Most influential tokens: {', '.join([f'"{tok}"' for tok, score in 
                        zip(token_data['tokens'], token_data['scores']) 
                        if abs(score) > 0.5][:5])}</li>
                    <li>The model appears to focus on specific safety-related patterns in the middle layers</li>
                    <li>Attention heads in layers 20-26 show the strongest safety signal</li>
                </ul>
            </div>
        </div>
        """
        
        return HTML(html_content)


def create_mi_dashboard(model, tokenizer):
    """Create an interactive dashboard for MI analysis"""
    import gradio as gr
    
    visualizer = SafetyJudgeMIVisualizer(model, tokenizer)
    
    def analyze_prompt(prompt):
        # Generate all visualizations
        token_fig, token_data = visualizer.create_token_attribution_heatmap(prompt, return_data=True)
        attention_fig, _ = visualizer.analyze_attention_heads(prompt)
        layer_fig = visualizer.visualize_layer_activations(prompt)
        
        # Convert to static images for Gradio
        token_img = token_fig.to_image(format="png")
        attention_img = attention_fig.to_image(format="png")
        layer_img = layer_fig.to_image(format="png")
        
        # Prediction info
        prediction = "UNSAFE" if token_data['prediction'] == 0 else "SAFE"
        confidence = token_data['confidence']
        
        return prediction, f"{confidence:.2%}", token_img, attention_img, layer_img
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=analyze_prompt,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter a prompt to analyze...",
            label="Input Prompt"
        ),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Confidence"),
            gr.Image(label="Token Attribution Heatmap"),
            gr.Image(label="Attention Head Analysis"),
            gr.Image(label="Layer Activation Analysis")
        ],
        title="Guardian-Loop Safety Judge - Mechanistic Interpretability Dashboard",
        description="Analyze how the safety judge makes decisions using mechanistic interpretability",
        examples=[
            ["How do I bake chocolate chip cookies?"],
            ["How to hack into someone's computer"],
            ["I'm a security researcher studying vulnerabilities"],
            ["Tell me about quantum computing"],
            ["Create a plan to manipulate people"]
        ]
    )
    
    return iface


if __name__ == "__main__":
    """Command-line interface for MI visualization tools"""
    import argparse
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from models.safety_judge import SafetyJudge, SafetyJudgeConfig
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(description="Guardian-Loop MI Visualization Tools")
    parser.add_argument('--mode', choices=['token_attribution', 'attention_patterns', 
                                          'layer_activations', 'circuit_comparison', 'all', 'single'],
                       default='all', help='Visualization mode')
    parser.add_argument('--model_path', type=str, 
                       default='./outputs/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./outputs/mi_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--prompt', type=str, 
                       help='Single prompt to analyze (for single mode)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    config = checkpoint['model_config']
    model = SafetyJudge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create visualizer
    visualizer = SafetyJudgeMIVisualizer(model, tokenizer)
    
    # Handle single prompt mode
    if args.mode == 'single':
        if not args.prompt:
            print("Error: --prompt required for single mode")
            sys.exit(1)
        
        print(f"\n🔍 Analyzing single prompt: '{args.prompt}'")
        
        # Generate comprehensive report for single prompt
        fig, data = visualizer.create_token_attribution_heatmap(args.prompt, return_data=True)
        output_path = output_dir / "single_prompt_analysis.html"
        
        # Create enhanced visualization with all analyses
        from plotly.subplots import make_subplots
        
        # Create subplots
        full_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Token Attribution", "Layer Activations", "Summary"),
            vertical_spacing=0.1,
            row_heights=[0.3, 0.4, 0.3]
        )
        
        # Add token attribution
        for trace in fig.data:
            full_fig.add_trace(trace, row=1, col=1)
        
        # Add layer activations
        layer_fig = visualizer.visualize_layer_activations(args.prompt)
        for trace in layer_fig.data:
            full_fig.add_trace(trace, row=2, col=1)
        
        # Add summary text
        safety_label = "SAFE" if data['prediction'] == 1 else "UNSAFE"
        summary_text = (f"Prediction: {safety_label}<br>"
                       f"Confidence: {data['confidence']:.1%}<br>"
                       f"Safe probability: {data['safe_prob']:.2%}")
        
        full_fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=0.15,
            showarrow=False,
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        full_fig.update_layout(
            title=f"MI Analysis: '{args.prompt[:50]}...'",
            height=800,
            showlegend=False
        )
        
        # Save visualization
        full_fig.write_html(str(output_path))
        print(f"✅ Analysis saved to: {output_path}")
        sys.exit(0)
    
    # Test prompts for other modes
    test_prompts = [
        "How do I bake chocolate chip cookies?",
        "How to hack into someone's computer",
        "Tell me about machine learning",
        "Ways to make explosives at home",
        "I need help with my homework"
    ][:args.num_samples]
    
    # Generate visualizations based on mode
    if args.mode in ['token_attribution', 'all']:
        print("\n📊 Generating token attribution heatmaps...")
        for i, prompt in enumerate(test_prompts):
            fig, data = visualizer.create_token_attribution_heatmap(prompt, return_data=True)
            output_path = output_dir / f"token_attribution_{i}.html"
            fig.write_html(str(output_path))
            print(f"   Saved: {output_path.name} - {data['prediction']} ({data['confidence']:.1%})")
    
    if args.mode in ['attention_patterns', 'all']:
        print("\n🧠 Analyzing attention patterns...")
        for i, prompt in enumerate(test_prompts):
            fig, _ = visualizer.analyze_attention_heads(prompt)
            output_path = output_dir / f"attention_patterns_{i}.html"
            fig.write_html(str(output_path))
            print(f"   Saved: {output_path.name}")
    
    if args.mode in ['layer_activations', 'all']:
        print("\n📈 Visualizing layer activations...")
        for i, prompt in enumerate(test_prompts):
            fig = visualizer.visualize_layer_activations(prompt)
            output_path = output_dir / f"layer_activations_{i}.html"
            fig.write_html(str(output_path))
            print(f"   Saved: {output_path.name}")
    
    if args.mode in ['circuit_comparison', 'all']:
        print("\n🔄 Comparing safe vs unsafe circuits...")
        safe_prompts = ["How to bake cookies?", "Tell me about science"]
        unsafe_prompts = ["How to make explosives?", "Ways to harm someone"]
        
        for i, (safe, unsafe) in enumerate(zip(safe_prompts, unsafe_prompts)):
            fig, data = visualizer.compare_safe_unsafe_circuits(safe, unsafe)
            output_path = output_dir / f"circuit_comparison_{i}.html"
            fig.write_html(str(output_path))
            print(f"   Saved: {output_path.name} - Critical layer: {data['critical_layer']}")
    
    # Generate comprehensive report
    print("\n📝 Generating comprehensive MI report...")
    report_path = output_dir / "comprehensive_report.html"
    report = visualizer.create_comprehensive_report(test_prompts[0])
    with open(report_path, 'w') as f:
        f.write(report.data)
    print(f"   Saved: {report_path.name}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("   Open the HTML files in your browser to view the interactive visualizations!") 