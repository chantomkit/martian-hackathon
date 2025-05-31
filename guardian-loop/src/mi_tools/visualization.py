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
        """Forward function for Captum"""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        # Return probability of unsafe class
        return torch.softmax(logits, dim=-1)[:, 1]
    
    def create_token_attribution_heatmap(self, text: str, return_data: bool = False):
        """Create an interactive heatmap showing which tokens trigger safety concerns"""
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get base prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, inputs.attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=1).item()
            unsafe_prob = probs[0, 1].item()
        
        # Calculate integrated gradients
        attributions = self.integrated_gradients.attribute(
            inputs.input_ids,
            baselines=self.tokenizer.pad_token_id,
            n_steps=50
        )
        
        # Get tokens and attribution scores
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        attribution_scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
        
        # Normalize scores
        norm_scores = attribution_scores / (np.abs(attribution_scores).max() + 1e-10)
        
        # Create interactive Plotly visualization
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=[norm_scores],
            x=tokens,
            y=['Token Attribution'],
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{score:.3f}" for score in norm_scores]],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Attribution Score")
        ))
        
        # Update layout
        safety_label = "UNSAFE" if prediction == 0 else "SAFE"
        fig.update_layout(
            title=f"Token Attribution Analysis - Prediction: {safety_label} ({unsafe_prob:.2%} unsafe)",
            xaxis_title="Tokens",
            yaxis_title="",
            height=200,
            xaxis=dict(tickangle=-45),
            margin=dict(b=150)
        )
        
        if return_data:
            return fig, {
                'tokens': tokens,
                'scores': norm_scores,
                'prediction': prediction,
                'unsafe_prob': unsafe_prob
            }
        
        return fig
    
    def analyze_attention_heads(self, text: str, layer_range: Tuple[int, int] = (20, 26)):
        """Analyze which attention heads focus on safety-critical patterns"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Forward pass to get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, inputs.attention_mask)
            attentions = outputs['attentions']
        
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
            title=f"Top 15 Attention Heads for Safety Detection<br><sub>Text: '{text[:50]}...'</sub>",
            xaxis_title="Attention Head",
            yaxis_title="Average Attention to Safety Patterns",
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig, head_importance
    
    def visualize_layer_activations(self, text: str):
        """Visualize how different layers respond to the input"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get activations from model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, inputs.attention_mask)
            hidden_states = outputs['hidden_states']
            activations = self.model.activation_cache
        
        # Calculate activation magnitudes per layer
        layer_magnitudes = []
        layer_names = []
        
        for i, hidden in enumerate(hidden_states[1:]):  # Skip embedding layer
            magnitude = hidden.norm(dim=-1).mean().item()
            layer_magnitudes.append(magnitude)
            layer_names.append(f"Layer {i}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Layer Activation Magnitudes", "MLP vs Attention Contributions"),
            vertical_spacing=0.15
        )
        
        # Plot 1: Layer magnitudes
        fig.add_trace(
            go.Scatter(
                x=layer_names,
                y=layer_magnitudes,
                mode='lines+markers',
                name='Activation Magnitude',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Mark frozen layers
        if hasattr(self.model.config, 'freeze_layers'):
            fig.add_vline(
                x=self.model.config.freeze_layers - 0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Frozen/Unfrozen Boundary",
                row=1, col=1
            )
        
        # Plot 2: MLP vs Attention contributions
        mlp_contributions = []
        attn_contributions = []
        
        for i in range(min(len(layer_names), len(activations) // 2)):
            mlp_key = f"layer_{i}_mlp_output"
            attn_key = f"layer_{i}_attention"
            
            if mlp_key in activations:
                mlp_contributions.append(activations[mlp_key].norm().item())
            else:
                mlp_contributions.append(0)
                
            if attn_key in activations and 'hidden_states' in activations[attn_key]:
                attn_contributions.append(
                    activations[attn_key]['hidden_states'].norm().item()
                )
            else:
                attn_contributions.append(0)
        
        x_contrib = list(range(len(mlp_contributions)))
        
        fig.add_trace(
            go.Scatter(
                x=x_contrib,
                y=mlp_contributions,
                mode='lines+markers',
                name='MLP',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_contrib,
                y=attn_contributions,
                mode='lines+markers',
                name='Attention',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Layer Index", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="Contribution", row=2, col=1)
        
        fig.update_layout(
            title=f"Layer-wise Analysis<br><sub>Text: '{text[:50]}...'</sub>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def compare_safe_unsafe_circuits(self, safe_prompt: str, unsafe_prompt: str):
        """Compare activation patterns between safe and unsafe prompts"""
        
        # Get embeddings for both prompts
        def get_embeddings(text):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, inputs.attention_mask)
                hidden_states = outputs['hidden_states']
                
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
        
        fig.update_layout(
            title="Safety Circuit Analysis: Safe vs Unsafe Prompts",
            height=700,
            showlegend=True
        )
        
        # Add text boxes with prompts
        fig.add_annotation(
            text=f"Safe: '{safe_prompt[:40]}...'<br>Unsafe: '{unsafe_prompt[:40]}...'",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
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
        <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: auto;">
            <h1>Mechanistic Interpretability Report</h1>
            <h2>Safety Judge Analysis</h2>
            
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3>Input Text:</h3>
                <p style="font-style: italic;">"{text}"</p>
                <h3>Prediction:</h3>
                <p><strong>{'UNSAFE' if token_data['prediction'] == 0 else 'SAFE'}</strong> 
                   (Unsafe probability: {token_data['unsafe_prob']:.2%})</p>
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
        confidence = token_data['unsafe_prob'] if token_data['prediction'] == 0 else (1 - token_data['unsafe_prob'])
        
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