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
    freeze_layers: int = 24  # Freeze first 24 layers (out of 32)
    probe_hidden_size: int = 256
    num_classes: int = 2  # Safe vs Unsafe
    dropout_rate: float = 0.1
    max_length: int = 512
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16


class SafetyProbeHead(nn.Module):
    """Lightweight probe head for safety classification"""
    
    def __init__(self, input_dim: int, config: SafetyJudgeConfig):
        super().__init__()
        self.config = config
        
        # Two-layer MLP with dropout
        self.probe = nn.Sequential(
            nn.Linear(input_dim, config.probe_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.probe_hidden_size, config.probe_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.probe_hidden_size // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize probe weights with small values"""
        for module in self.probe.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through probe head"""
        # Pool hidden states (use CLS token or mean pooling)
        if hidden_states.ndim == 3:  # [batch, seq_len, hidden]
            pooled = hidden_states[:, 0, :]  # Use first token (CLS-like)
        else:
            pooled = hidden_states
        
        return self.probe(pooled)


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
        
        # Extract hidden states from the last unfrozen layer
        hidden_states = outputs.hidden_states[self.config.freeze_layers]
        
        # Pass through probe head
        logits = self.probe_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
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