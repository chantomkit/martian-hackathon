# Mechanistic Interpretability Implementation Guide
## Concrete, Feasible Visualizations for Guardian-Loop

### 1. Essential Libraries Setup
```python
# Core dependencies
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
import plotly.graph_objects as go
from captum.attr import IntegratedGradients, LayerConductance
import pandas as pd

# Mechanistic interpretability specific
from fancy_einsum import einsum
from circuitsvis.attention import attention_patterns
from IPython.display import HTML
```

### 2. Token Attribution Heatmap (Day 1 - Easy Win)
```python
def create_token_heatmap(text, model, tokenizer):
    """Create an interactive heatmap showing which tokens trigger safety concerns"""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Get attributions using integrated gradients
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        inputs.embeds, 
        target=1,  # Unsafe class
        return_convergence_delta=True
    )
    
    # Aggregate attribution scores
    token_scores = attributions.sum(dim=-1).squeeze(0)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(len(tokens)*0.8, 2))
    
    # Normalize scores for coloring
    norm_scores = (token_scores - token_scores.min()) / (token_scores.max() - token_scores.min())
    colors = plt.cm.Reds(norm_scores)
    
    # Create bars
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, [1]*len(tokens), color=colors)
    
    # Add token labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Safety Concern Level')
    ax.set_title(f'Token Attribution: "{text}"')
    
    # Add score annotations
    for i, score in enumerate(token_scores):
        ax.text(0.5, i, f'{score:.2f}', ha='center', va='center')
    
    plt.tight_layout()
    return fig, token_scores
```

### 3. Attention Head Analysis (Day 2 - Medium Complexity)
```python
def analyze_safety_heads(model, text, tokenizer, layer_range=(4, 6)):
    """Identify which attention heads focus on safety-critical patterns"""
    
    # Hook storage
    attention_weights = {}
    
    def store_attn_weights(name):
        def hook(module, input, output):
            attention_weights[name] = output[1]  # attention weights
        return hook
    
    # Register hooks
    handles = []
    for layer in range(*layer_range):
        handle = model.transformer.h[layer].attn.register_forward_hook(
            store_attn_weights(f'layer_{layer}')
        )
        handles.append(handle)
    
    # Forward pass
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    # Analyze patterns
    safety_tokens = identify_safety_tokens(text, tokenizer)
    head_importance = {}
    
    for layer_name, attn in attention_weights.items():
        # attn shape: [batch, heads, seq, seq]
        n_heads = attn.shape[1]
        
        for head in range(n_heads):
            # Check if head attends to safety tokens
            attn_to_safety = attn[0, head, :, safety_tokens].mean()
            head_importance[f"{layer_name}_head_{head}"] = attn_to_safety.item()
    
    # Create visualization
    fig = go.Figure()
    
    # Sort heads by importance
    sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig.add_trace(go.Bar(
        x=[h[0] for h in sorted_heads],
        y=[h[1] for h in sorted_heads],
        text=[f"{h[1]:.3f}" for h in sorted_heads],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 10 Attention Heads for Safety Detection",
        xaxis_title="Head",
        yaxis_title="Average Attention to Safety Tokens",
        xaxis_tickangle=-45
    )
    
    return fig, head_importance
```

### 4. Safety Circuit Visualization (Day 2-3)
```python
def visualize_safety_circuit(model, unsafe_prompt, safe_prompt, tokenizer):
    """Show how different prompts activate different paths through the model"""
    
    # Get embeddings at each layer
    def get_layer_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt")
        embeddings = []
        
        def hook(module, input, output):
            embeddings.append(output[0].mean(dim=1))  # Average over sequence
        
        handles = []
        for layer in model.transformer.h:
            handle = layer.register_forward_hook(hook)
            handles.append(handle)
        
        with torch.no_grad():
            model(**inputs)
        
        for handle in handles:
            handle.remove()
            
        return torch.stack(embeddings)
    
    unsafe_embeds = get_layer_embeddings(unsafe_prompt)
    safe_embeds = get_layer_embeddings(safe_prompt)
    
    # Calculate divergence between paths
    divergence = torch.norm(unsafe_embeds - safe_embeds, dim=-1)
    
    # Create flow diagram
    fig = go.Figure()
    
    # Add trace for divergence
    fig.add_trace(go.Scatter(
        x=list(range(len(divergence))),
        y=divergence.cpu().numpy(),
        mode='lines+markers',
        name='Path Divergence',
        line=dict(width=3)
    ))
    
    # Mark critical layers
    critical_layer = divergence.argmax().item()
    fig.add_vline(x=critical_layer, line_dash="dash", 
                  annotation_text=f"Critical Layer {critical_layer}")
    
    fig.update_layout(
        title="Safety Circuit Activation Path",
        xaxis_title="Layer",
        yaxis_title="Embedding Divergence",
        showlegend=True
    )
    
    return fig, critical_layer
```

### 5. Live Adversarial Probe (Day 3 - Interactive Demo)
```python
def create_interactive_adversarial_probe(judge_model, tokenizer):
    """Interactive widget to test prompts and see MI analysis in real-time"""
    
    from ipywidgets import interact, widgets, Output
    import matplotlib.pyplot as plt
    
    output = Output()
    
    def analyze_prompt(prompt):
        with output:
            output.clear_output(wait=True)
            
            # Get prediction
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                logits = judge_model(**inputs).logits
                pred = torch.softmax(logits, dim=-1)
                is_safe = pred[0, 0] > pred[0, 1]
            
            # Create dashboard
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. Prediction bar
            ax1 = axes[0, 0]
            ax1.bar(['Safe', 'Unsafe'], pred[0].cpu())
            ax1.set_title(f'Prediction: {"SAFE" if is_safe else "UNSAFE"}')
            ax1.set_ylim(0, 1)
            
            # 2. Token heatmap
            ax2 = axes[0, 1]
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            token_scores = get_token_importance(judge_model, inputs)
            
            im = ax2.imshow(token_scores.reshape(1, -1), cmap='Reds', aspect='auto')
            ax2.set_xticks(range(len(tokens)))
            ax2.set_xticklabels(tokens, rotation=45, ha='right')
            ax2.set_title('Token Importance')
            plt.colorbar(im, ax=ax2)
            
            # 3. Layer activation
            ax3 = axes[1, 0]
            layer_acts = get_layer_activations(judge_model, inputs)
            ax3.plot(layer_acts)
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Activation Magnitude')
            ax3.set_title('Layer-wise Activation')
            
            # 4. Suggested mutations
            ax4 = axes[1, 1]
            ax4.axis('off')
            mutations = suggest_adversarial_mutations(prompt, token_scores)
            ax4.text(0.1, 0.9, "Try these mutations:", transform=ax4.transAxes, 
                    fontsize=12, fontweight='bold')
            for i, mut in enumerate(mutations[:3]):
                ax4.text(0.1, 0.7-i*0.2, f"{i+1}. {mut}", transform=ax4.transAxes,
                        fontsize=10, wrap=True)
            
            plt.tight_layout()
            plt.show()
    
    # Create interactive widget
    prompt_input = widgets.Textarea(
        value='How do I pick a lock?',
        placeholder='Enter prompt to analyze',
        description='Prompt:',
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    interact(analyze_prompt, prompt=prompt_input)
    display(output)
```

### 6. Before/After LoRA Update Comparison
```python
def compare_before_after_lora(judge_v1, judge_v2, test_prompts, tokenizer):
    """Show how LoRA update changed the model's decision boundaries"""
    
    results = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Get predictions from both versions
        with torch.no_grad():
            pred_v1 = torch.softmax(judge_v1(**inputs).logits, dim=-1)
            pred_v2 = torch.softmax(judge_v2(**inputs).logits, dim=-1)
        
        # Get attention patterns
        attn_v1 = get_attention_patterns(judge_v1, inputs)
        attn_v2 = get_attention_patterns(judge_v2, inputs)
        
        # Calculate changes
        pred_change = (pred_v2 - pred_v1)[0, 1].item()  # Change in unsafe score
        attn_change = torch.norm(attn_v2 - attn_v1).item()
        
        results.append({
            'prompt': prompt[:50] + '...',
            'pred_change': pred_change,
            'attn_change': attn_change,
            'flipped': (pred_v1[0, 0] > 0.5) != (pred_v2[0, 0] > 0.5)
        })
    
    # Create comparison visualization
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prediction changes
    colors = ['red' if x else 'blue' for x in df['flipped']]
    ax1.barh(df.index, df['pred_change'], color=colors)
    ax1.set_xlabel('Change in Unsafe Score')
    ax1.set_title('Prediction Changes After LoRA Update')
    ax1.set_yticks(df.index)
    ax1.set_yticklabels(df['prompt'], fontsize=8)
    
    # Attention pattern changes
    ax2.scatter(df['pred_change'], df['attn_change'], c=colors)
    ax2.set_xlabel('Prediction Change')
    ax2.set_ylabel('Attention Pattern Change')
    ax2.set_title('Correlation: Prediction vs Attention Changes')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Flipped'),
                      Patch(facecolor='blue', label='Not Flipped')]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    return fig, df
```

### 7. Quick Implementation Tips

#### For Limited Time:
1. **Start with token heatmaps** - Most visual, easiest to implement
2. **Use existing tools** - Captum for attributions, circuitsvis for attention
3. **Pre-compute expensive operations** - Cache embeddings and attention patterns

#### For GPU Efficiency:
```python
# Batch process multiple prompts
def batch_analyze(prompts, model, tokenizer, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        results.extend(process_batch_outputs(outputs))
    return results
```

#### For Demo Impact:
1. **Focus on surprising findings** - Show cases where the judge ignores obvious safety words
2. **Make it interactive** - Let judges test their own prompts
3. **Show the fix working** - Before/after LoRA update on the same adversarial example

This guide provides concrete, implementable visualizations that can be built within your 72-hour timeframe while demonstrating deep mechanistic understanding of your safety judge. 