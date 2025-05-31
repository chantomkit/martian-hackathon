"""
Safety Judge Model for Guardian-Loop
Uses Llama 3.1 8B with frozen layers and a lightweight probe head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, LlamaForSequenceClassification
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SafetyJudgeConfig:
    """Configuration for the Safety Judge model"""
    base_model: str = "meta-llama/Llama-3.1-8B"
    freeze_layers: int = 20  # Reduced from 24 - unfreeze last 12 layers for more capacity
    probe_hidden_size: int = 128  # Reduced from 256 to prevent overfitting
    num_classes: int = 2  # Safe vs Unsafe
    dropout_rate: float = 0.2  # Increased from 0.1 for better regularization
    max_length: int = 512
    use_pooler: str = "mean"  # "mean", "max", or "cls" pooling
    use_layer_norm: bool = True  # Add layer norm for stability


class SafetyProbeHead(nn.Module):
    """Lightweight probe head for safety classification"""
    
    def __init__(self, input_dim: int, config: SafetyJudgeConfig):
        super().__init__()
        self.config = config
        
        # Simpler 2-layer MLP for 8K samples
        layers = []
        
        # Optional layer norm for stability
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, config.probe_hidden_size),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(config.dropout_rate),
        ])
        
        # Output layer
        layers.append(nn.Linear(config.probe_hidden_size, config.num_classes))
        
        self.probe = nn.Sequential(*layers)
        
        # Initialize weights with careful scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize probe weights with small values for stable training"""
        for module in self.probe.modules():
            if isinstance(module, nn.Linear):
                # Smaller initialization for stability with limited data
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through probe head"""
        return self.probe(hidden_states)


class SafetyJudge(nn.Module):
    """Main Safety Judge model combining Llama 3.1 with probe head"""
    
    def __init__(self, config: SafetyJudgeConfig):
        super().__init__()
        self.config = config
        
        # Load base Llama model
        self.base_model = AutoModel.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Freeze specified layers
        self._freeze_base_layers()
        
        # Get hidden size from base model
        hidden_size = self.base_model.config.hidden_size
        
        # Add probe head
        self.probe_head = SafetyProbeHead(hidden_size, config)
        
        # For mechanistic interpretability - store activations
        self.activation_cache = {}
        self.register_hooks()
        
    def _freeze_base_layers(self):
        """Freeze the first N layers of the base model"""
        # Freeze embeddings
        for param in self.base_model.embed_tokens.parameters():
            param.requires_grad = False
        
        # Freeze specified transformer layers
        for i in range(self.config.freeze_layers):
            for param in self.base_model.layers[i].parameters():
                param.requires_grad = False
    
    def register_hooks(self):
        """Register forward hooks for mechanistic interpretability"""
        self.handles = []
        
        # Hook into attention layers for MI analysis
        for i, layer in enumerate(self.base_model.layers):
            # Attention outputs
            handle = layer.self_attn.register_forward_hook(
                self._create_attention_hook(f"layer_{i}_attention")
            )
            self.handles.append(handle)
            
            # MLP outputs
            handle = layer.mlp.register_forward_hook(
                self._create_activation_hook(f"layer_{i}_mlp")
            )
            self.handles.append(handle)
    
    def _create_attention_hook(self, name: str):
        """Create hook for storing attention patterns"""
        def hook(module, input, output):
            # Store attention weights if available
            if isinstance(output, tuple) and len(output) > 1:
                self.activation_cache[name] = {
                    'attention_weights': output[1].detach().cpu() if output[1] is not None else None,
                    'hidden_states': output[0].detach().cpu()
                }
        return hook
    
    def _create_activation_hook(self, name: str):
        """Create hook for storing MLP activations"""
        def hook(module, input, output):
            self.activation_cache[f"{name}_output"] = output.detach().cpu()
        return hook
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the safety judge"""
        
        # Clear activation cache
        self.activation_cache.clear()
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Get hidden states from multiple layers for better representation
        # Use last 3 unfrozen layers and combine them
        layer_outputs = []
        for i in range(self.config.freeze_layers, min(self.config.freeze_layers + 3, len(outputs.hidden_states) - 1)):
            layer_outputs.append(outputs.hidden_states[i])
        
        # Combine layers (weighted average)
        if len(layer_outputs) > 1:
            hidden_states = torch.stack(layer_outputs).mean(dim=0)
        else:
            hidden_states = outputs.hidden_states[self.config.freeze_layers]
        
        # Apply pooling strategy
        if self.config.use_pooler == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.config.use_pooler == "max":
            # Max pooling
            pooled = hidden_states.max(dim=1)[0]
        else:  # "cls"
            # Use first token
            pooled = hidden_states[:, 0, :]
        
        # Pass through probe head
        logits = self.probe_head(pooled)
        
        if return_dict:
            return {
                'logits': logits,
                'pooled': pooled,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions,
                'activations': self.activation_cache.copy()
            }
        
        return logits
    
    def predict(self, text: str, tokenizer: AutoTokenizer) -> Tuple[bool, float]:
        """Predict if a prompt is safe or unsafe"""
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.base_model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Safe = 0, Unsafe = 1
            unsafe_prob = probs[0, 1].item()
            is_safe = unsafe_prob < 0.5
            
        return is_safe, 1 - unsafe_prob
    
    def get_token_importance(self, text: str, tokenizer: AutoTokenizer) -> np.ndarray:
        """Get importance scores for each token (for MI visualization)"""
        # This will be implemented with integrated gradients
        # Placeholder for now
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        return np.random.rand(inputs.input_ids.shape[1])
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda"):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_safety_judge(config: Optional[SafetyJudgeConfig] = None) -> SafetyJudge:
    """Factory function to create a safety judge"""
    if config is None:
        config = SafetyJudgeConfig()
    
    return SafetyJudge(config) 