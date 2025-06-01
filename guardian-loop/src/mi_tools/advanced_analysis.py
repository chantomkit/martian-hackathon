"""
Advanced Mechanistic Interpretability Analysis for Guardian-Loop
State-of-the-art techniques for understanding safety judge decisions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SafetyNeuron:
    """Represents a neuron that responds to safety-related features"""
    layer: int
    neuron_idx: int
    activation_mean: float
    activation_std: float
    safe_vs_unsafe_difference: float
    top_activating_tokens: List[str]
    safety_keywords_correlation: float


@dataclass 
class SafetyCircuit:
    """Represents a computational circuit for safety detection"""
    neurons: List[SafetyNeuron]
    circuit_strength: float
    activated_by: List[str]  # Types of unsafe content
    layer_flow: Dict[int, List[int]]  # Layer -> neuron indices


class AdvancedSafetyAnalyzer:
    """Advanced mechanistic interpretability for the safety judge"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Cache for neuron activations
        self.neuron_cache = {}
        self.safety_neurons = []
        self.safety_circuits = []
        
        # Register hooks for detailed analysis
        self._register_detailed_hooks()
        
    def _register_detailed_hooks(self):
        """Register hooks to capture detailed neuron activations"""
        self.handles = []
        
        for idx, layer in enumerate(self.model.base_model.model.layers):
            # Hook into MLP activations for neuron-level analysis
            handle = layer.mlp.act_fn.register_forward_hook(
                self._create_neuron_hook(f"layer_{idx}_neurons")
            )
            self.handles.append(handle)
            
            # Hook into attention outputs
            handle = layer.self_attn.register_forward_hook(
                self._create_attention_hook(f"layer_{idx}_attention")
            )
            self.handles.append(handle)
    
    def _create_neuron_hook(self, name: str):
        def hook(module, input, output):
            # Store individual neuron activations
            self.neuron_cache[name] = output.detach().cpu()
        return hook
    
    def _create_attention_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.neuron_cache[name] = output[0].detach().cpu()
        return hook
    
    def identify_safety_neurons(self, safe_prompts: List[str], unsafe_prompts: List[str]) -> List[SafetyNeuron]:
        """Identify neurons that differentially activate for safe vs unsafe content"""
        print("🔬 Identifying safety-critical neurons...")
        
        # Limit number of prompts for efficiency
        max_prompts = 5  # Further reduced for speed
        safe_prompts = safe_prompts[:max_prompts]
        unsafe_prompts = unsafe_prompts[:max_prompts]
        print(f"   Analyzing {len(safe_prompts)} safe and {len(unsafe_prompts)} unsafe prompts...")
        
        # Collect activations for safe prompts
        print("   Processing safe prompts...")
        safe_activations = self._collect_neuron_activations(safe_prompts, show_progress=True)
        
        # Collect activations for unsafe prompts  
        print("   Processing unsafe prompts...")
        unsafe_activations = self._collect_neuron_activations(unsafe_prompts, show_progress=True)
        
        # Find neurons with significant differences
        print("   Analyzing neuron activations...")
        safety_neurons = []
        
        for layer_name in safe_activations:
            if 'neurons' not in layer_name:
                continue
                
            layer_idx = int(layer_name.split('_')[1])
            safe_acts = safe_activations[layer_name]
            unsafe_acts = unsafe_activations[layer_name]
            
            # Calculate neuron-wise statistics
            safe_mean = safe_acts.mean(dim=(0, 1))  # Average over batch and sequence
            unsafe_mean = unsafe_acts.mean(dim=(0, 1))
            
            # Find neurons with large activation differences
            diff = (unsafe_mean - safe_mean).abs()
            threshold = diff.mean() + 2 * diff.std()  # 2 standard deviations
            
            significant_neurons = torch.where(diff > threshold)[0]
            
            # Limit neurons per layer for efficiency
            significant_neurons = significant_neurons[:3]  # Reduced to 3 neurons per layer
            
            for neuron_idx in significant_neurons:
                neuron = SafetyNeuron(
                    layer=layer_idx,
                    neuron_idx=neuron_idx.item(),
                    activation_mean=unsafe_mean[neuron_idx].item(),
                    activation_std=unsafe_acts[:, :, neuron_idx].std().item(),
                    safe_vs_unsafe_difference=diff[neuron_idx].item(),
                    top_activating_tokens=self._find_top_activating_tokens(
                        unsafe_prompts[:2],  # Only use 2 prompts for token analysis
                        layer_name, 
                        neuron_idx
                    ),
                    safety_keywords_correlation=self._calculate_keyword_correlation(
                        unsafe_acts[:, :, neuron_idx], unsafe_prompts
                    )
                )
                safety_neurons.append(neuron)
        
        # Sort by importance
        safety_neurons.sort(key=lambda n: n.safe_vs_unsafe_difference, reverse=True)
        self.safety_neurons = safety_neurons[:30]  # Reduced to top 30 neurons
        
        print(f"   Found {len(self.safety_neurons)} safety-critical neurons")
        return self.safety_neurons
    
    def _collect_neuron_activations(self, prompts: List[str], show_progress: bool = False) -> Dict[str, torch.Tensor]:
        """Collect neuron activations for a set of prompts"""
        all_activations = {}
        max_length = 0
        
        # Process in batches for efficiency
        batch_size = 2  # Small batch to avoid OOM
        temp_activations = []
        
        # Create progress bar if requested
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="      ", total=len(prompts)//batch_size + 1)
        
        for i in iterator:
            batch_prompts = prompts[i:i+batch_size]
            self.neuron_cache.clear()
            
            # Format and tokenize batch
            formatted = [self.model.config.prompt_template.format(text=p) for p in batch_prompts]
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_length,
                padding=True
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(inputs.input_ids, inputs.attention_mask)
            
            # Store activations for this batch
            for b_idx in range(len(batch_prompts)):
                prompt_acts = {}
                for name, acts in self.neuron_cache.items():
                    # Extract single prompt activations from batch
                    prompt_acts[name] = acts[b_idx:b_idx+1]  # Keep batch dimension
                    if acts.shape[1] > max_length:
                        max_length = acts.shape[1]
                
                temp_activations.append(prompt_acts)
            
            # Clear GPU memory
            del inputs
            torch.cuda.empty_cache()
        
        # Pad all activations to max length and aggregate
        for prompt_acts in temp_activations:
            for name, acts in prompt_acts.items():
                # Pad sequence dimension if needed
                if acts.shape[1] < max_length:
                    pad_size = max_length - acts.shape[1]
                    padded = torch.nn.functional.pad(acts, (0, 0, 0, pad_size), value=0)
                else:
                    padded = acts
                
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(padded)
        
        # Stack activations
        for name in all_activations:
            all_activations[name] = torch.cat(all_activations[name], dim=0)
        
        return all_activations
    
    def _find_top_activating_tokens(self, prompts: List[str], layer_name: str, neuron_idx: int) -> List[str]:
        """Find tokens that most activate a specific neuron"""
        top_tokens = []
        max_activations = []
        
        for prompt in prompts[:10]:  # Sample for efficiency
            self.neuron_cache.clear()
            
            formatted = self.model.config.prompt_template.format(text=prompt)
            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True).to(self.device)
            
            with torch.no_grad():
                _ = self.model(inputs.input_ids, inputs.attention_mask)
            
            if layer_name in self.neuron_cache:
                acts = self.neuron_cache[layer_name][0, :, neuron_idx]  # [seq_len]
                # Get top 5 activations per prompt instead of just 1
                top_indices = acts.topk(min(5, len(acts))).indices
                
                tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                for idx in top_indices:
                    if idx < len(tokens):
                        top_tokens.append(tokens[idx])
                        max_activations.append(acts[idx].item())
        
        # Return unique tokens sorted by activation
        token_acts = {}
        for token, act in zip(top_tokens, max_activations):
            if token not in token_acts or act > token_acts[token]:
                token_acts[token] = act
        
        # Return top 10 tokens instead of 5
        return sorted(token_acts.keys(), key=lambda t: token_acts[t], reverse=True)[:10]
    
    def _calculate_keyword_correlation(self, activations: torch.Tensor, prompts: List[str]) -> float:
        """Calculate correlation between neuron activation and safety keywords"""
        safety_keywords = ['harm', 'dangerous', 'illegal', 'weapon', 'drug', 'kill', 
                          'steal', 'hack', 'exploit', 'manipulate', 'bomb', 'violence',
                          'attack', 'threat', 'abuse', 'toxic']
        
        keyword_scores = []
        mean_activations = []
        
        for i, prompt in enumerate(prompts):
            # Count safety keywords
            prompt_lower = prompt.lower()
            keyword_count = sum(1 for kw in safety_keywords if kw in prompt_lower)
            keyword_scores.append(keyword_count)
            
            # Get mean activation for this prompt
            mean_activations.append(activations[i].mean().item())
        
        # Calculate correlation
        if len(set(keyword_scores)) > 1:  # Need variance
            correlation = np.corrcoef(keyword_scores, mean_activations)[0, 1]
            return abs(correlation)
        return 0.0
    
    def trace_safety_circuits(self) -> List[SafetyCircuit]:
        """Trace computational circuits involved in safety detection"""
        print("🔌 Tracing safety detection circuits...")
        
        if not self.safety_neurons:
            print("   No safety neurons identified yet!")
            return []
        
        # Group neurons by layer
        layer_neurons = {}
        for neuron in self.safety_neurons:
            if neuron.layer not in layer_neurons:
                layer_neurons[neuron.layer] = []
            layer_neurons[neuron.layer].append(neuron)
        
        # Build circuits by tracing activation flow
        circuits = []
        
        # Simple circuit: consecutive layers with high correlation
        sorted_layers = sorted(layer_neurons.keys())
        
        for i in range(len(sorted_layers) - 2):
            # Check if we have neurons in consecutive layers
            if sorted_layers[i+1] == sorted_layers[i] + 1:
                circuit_neurons = []
                circuit_neurons.extend(layer_neurons[sorted_layers[i]][:3])
                circuit_neurons.extend(layer_neurons[sorted_layers[i+1]][:3])
                
                if len(circuit_neurons) >= 4:
                    circuit = SafetyCircuit(
                        neurons=circuit_neurons,
                        circuit_strength=np.mean([n.safe_vs_unsafe_difference for n in circuit_neurons]),
                        activated_by=self._infer_activation_patterns(circuit_neurons),
                        layer_flow={
                            sorted_layers[i]: [n.neuron_idx for n in circuit_neurons if n.layer == sorted_layers[i]],
                            sorted_layers[i+1]: [n.neuron_idx for n in circuit_neurons if n.layer == sorted_layers[i+1]]
                        }
                    )
                    circuits.append(circuit)
        
        self.safety_circuits = sorted(circuits, key=lambda c: c.circuit_strength, reverse=True)[:5]
        print(f"   Found {len(self.safety_circuits)} safety circuits")
        
        return self.safety_circuits
    
    def _infer_activation_patterns(self, neurons: List[SafetyNeuron]) -> List[str]:
        """Infer what types of unsafe content activate this circuit"""
        all_tokens = []
        for neuron in neurons:
            all_tokens.extend(neuron.top_activating_tokens)
        
        # Categorize tokens
        categories = []
        
        violence_tokens = ['harm', 'kill', 'attack', 'weapon', 'bomb', 'hurt']
        if any(any(vt in token.lower() for vt in violence_tokens) for token in all_tokens):
            categories.append("violence/harm")
        
        illegal_tokens = ['illegal', 'hack', 'steal', 'drug', 'crime']
        if any(any(it in token.lower() for it in illegal_tokens) for token in all_tokens):
            categories.append("illegal activities")
        
        manipulation_tokens = ['manipulate', 'exploit', 'trick', 'deceive']
        if any(any(mt in token.lower() for mt in manipulation_tokens) for token in all_tokens):
            categories.append("manipulation/deception")
        
        if not categories:
            categories.append("general unsafe content")
        
        return categories
    
    def create_neuron_activation_map(self) -> go.Figure:
        """Create an interactive map of safety neuron activations"""
        if not self.safety_neurons:
            return go.Figure().add_annotation(text="No safety neurons identified yet")
        
        # Prepare data for visualization
        layers = [n.layer for n in self.safety_neurons]
        neuron_indices = [n.neuron_idx for n in self.safety_neurons]
        differences = [n.safe_vs_unsafe_difference for n in self.safety_neurons]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Scale the differences for better visibility
        scaled_differences = np.array(differences)
        min_diff = scaled_differences.min()
        max_diff = scaled_differences.max()
        if max_diff > min_diff:
            scaled_sizes = 20 + 80 * (scaled_differences - min_diff) / (max_diff - min_diff)
        else:
            scaled_sizes = [50] * len(differences)
        
        fig.add_trace(go.Scatter(
            x=layers,
            y=neuron_indices,
            mode='markers',
            marker=dict(
                size=scaled_sizes,  # Scaled for better visibility
                color=differences,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title="Safety Signal<br>Strength",
                    titleside="right",
                    tickmode="linear",
                    tick0=min(differences),
                    dtick=(max(differences) - min(differences)) / 5
                )
            ),
            text=[f"Layer {n.layer}, Neuron {n.neuron_idx}<br>" +
                  f"Difference: {n.safe_vs_unsafe_difference:.3f}<br>" +
                  f"Top tokens: {', '.join(n.top_activating_tokens[:5])}"  # Show 5 tokens for space
                  for n in self.safety_neurons],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Safety Neuron Activation Map",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                title="Layer",
                tickmode='linear',
                tick0=min(layers) if layers else 0,
                dtick=1
            ),
            yaxis_title="Neuron Index",
            height=700,
            width=1000,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            margin=dict(t=100, b=80, l=80, r=150),  # More margin for colorbar
            hovermode='closest'
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_circuit_diagram(self) -> go.Figure:
        """Create a network diagram of safety circuits"""
        if not self.safety_circuits:
            return go.Figure().add_annotation(text="No circuits traced yet")
        
        # Build network graph
        G = nx.DiGraph()
        
        # Add nodes and edges for each circuit
        for i, circuit in enumerate(self.safety_circuits):
            for layer, neurons in circuit.layer_flow.items():
                for neuron_idx in neurons:
                    node_id = f"L{layer}_N{neuron_idx}"
                    G.add_node(node_id, layer=layer, neuron=neuron_idx, circuit=i)
            
            # Add edges between consecutive layers
            layers = sorted(circuit.layer_flow.keys())
            for j in range(len(layers) - 1):
                for n1 in circuit.layer_flow[layers[j]]:
                    for n2 in circuit.layer_flow[layers[j+1]]:
                        G.add_edge(f"L{layers[j]}_N{n1}", f"L{layers[j+1]}_N{n2}", circuit=i)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        for trace in edge_trace:
            fig.add_trace(trace)
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(G.nodes[node]['circuit'])
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Circuit ID")
            ),
            text=node_text,
            textposition="top center",
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Safety Detection Circuit Diagram",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        
        return fig
    
    def analyze_failure_modes(self, failed_prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
        """Analyze why the model failed on certain prompts"""
        print("🔍 Analyzing failure modes...")
        
        failure_analysis = {
            'total_failures': len(failed_prompts),
            'failure_patterns': {},
            'missing_activations': [],
            'recommendations': []
        }
        
        for prompt, true_label, pred_label in failed_prompts[:20]:  # Analyze sample
            # Get neuron activations
            self.neuron_cache.clear()
            
            formatted = self.model.config.prompt_template.format(text=prompt)
            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, inputs.attention_mask)
            
            # Check if safety neurons activated
            activated_neurons = 0
            for neuron in self.safety_neurons[:10]:
                layer_name = f"layer_{neuron.layer}_neurons"
                if layer_name in self.neuron_cache:
                    activation = self.neuron_cache[layer_name][0, :, neuron.neuron_idx].max().item()
                    if activation > neuron.activation_mean:
                        activated_neurons += 1
            
            # Categorize failure
            if true_label == 0 and pred_label == 1:  # False negative (missed unsafe)
                if activated_neurons < 3:
                    failure_analysis['missing_activations'].append(prompt)  # Store full prompt
            
        return failure_analysis
    
    def export_analysis(self, output_dir: Path):
        """Export complete analysis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export neuron data
        neuron_data = []
        for neuron in self.safety_neurons:
            neuron_data.append({
                'layer': neuron.layer,
                'neuron_idx': neuron.neuron_idx,
                'activation_mean': neuron.activation_mean,
                'activation_std': neuron.activation_std,
                'safe_vs_unsafe_difference': neuron.safe_vs_unsafe_difference,
                'top_activating_tokens': neuron.top_activating_tokens,
                'safety_keywords_correlation': neuron.safety_keywords_correlation
            })
        
        with open(output_dir / 'safety_neurons.json', 'w') as f:
            json.dump(neuron_data, f, indent=2)
        
        # Export circuit data
        circuit_data = []
        for circuit in self.safety_circuits:
            circuit_data.append({
                'circuit_strength': circuit.circuit_strength,
                'activated_by': circuit.activated_by,
                'layer_flow': circuit.layer_flow,
                'num_neurons': len(circuit.neurons)
            })
        
        with open(output_dir / 'safety_circuits.json', 'w') as f:
            json.dump(circuit_data, f, indent=2)
        
        print(f"📁 Analysis exported to {output_dir}")


def create_training_evolution_visualization(checkpoints_dir: Path) -> go.Figure:
    """Visualize how safety detection evolves during training"""
    
    # This would load checkpoints and analyze how neurons/circuits develop
    # For now, create a placeholder
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Neuron Specialization", "Circuit Formation",
                       "Safety Detection Accuracy", "Feature Evolution"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Placeholder data
    epochs = list(range(1, 16))
    
    # Neuron specialization
    fig.add_trace(
        go.Scatter(x=epochs, y=np.cumsum(np.random.randn(15)) + 10,
                  name="Safety Neurons", line=dict(color='red')),
        row=1, col=1
    )
    
    # Circuit formation
    fig.add_trace(
        go.Scatter(x=epochs, y=np.cumsum(np.random.randn(15)) + 5,
                  name="Active Circuits", line=dict(color='blue')),
        row=1, col=2
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=0.5 + 0.4 * (1 - np.exp(-np.array(epochs)/5)),
                  name="Safety Detection", line=dict(color='green')),
        row=2, col=1
    )
    
    # Feature evolution
    fig.add_trace(
        go.Scatter(x=epochs, y=np.cumsum(np.random.randn(15)**2),
                  name="Feature Complexity", line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title="Safety Detection Evolution During Training",
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      font=dict(color='black'))
    
    return fig 