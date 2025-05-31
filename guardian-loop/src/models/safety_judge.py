"""
Safety Judge Model for Guardian-Loop
Uses Llama 3.1 8B with direct prompting for True/False outputs
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SafetyJudgeConfig:
    """Configuration for the Safety Judge model"""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    freeze_layers: int = 20  # Reduced from 24 - unfreeze last 12 layers for more capacity
    max_length: int = 512
    prompt_template: str = "Is the following text safe and appropriate? Answer with True or False only.\n\nText: {text}\nAnswer: "


class SafetyJudge(nn.Module):
    """Safety Judge model using direct prompting"""
    
    def __init__(self, config: SafetyJudgeConfig):
        super().__init__()
        self.config = config
        
        # Load base Llama model for generation
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Store reference to underlying model for easier access
        self.transformer: PreTrainedModel = self.base_model.model
        
        # Freeze specified layers
        self._freeze_base_layers()
        
        # For mechanistic interpretability - store activations
        self.activation_cache = {}
        self.register_hooks()
        
    def _freeze_base_layers(self):
        """Freeze the first N layers of the base model"""
        # Freeze embeddings
        for param in self.transformer.embed_tokens.parameters():
            param.requires_grad = False
        
        # Freeze specified transformer layers
        for i in range(self.config.freeze_layers):
            for param in self.transformer.layers[i].parameters():
                param.requires_grad = False
    
    def register_hooks(self):
        """Register forward hooks for mechanistic interpretability"""
        self.handles = []
        
        # Hook into attention layers for MI analysis
        for i, layer in enumerate(self.transformer.layers):
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
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """Forward pass through the safety judge"""
        
        # Clear activation cache
        self.activation_cache.clear()
        
        # Get model outputs
        outputs: CausalLMOutputWithPast = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Get logits for True/False tokens
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        
        if return_dict:
            return {
                'logits': logits,
                'loss': outputs.loss,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions,
                'activations': self.activation_cache.copy()
            }
        
        return logits
    
    def prepare_prompt(self, text: str) -> str:
        """Prepare the prompt for safety classification"""
        return self.config.prompt_template.format(text=text)
    
    def predict(self, text: str, tokenizer: AutoTokenizer) -> Tuple[bool, float]:
        """Predict if a prompt is safe or unsafe"""
        # Prepare prompt
        prompt = self.prepare_prompt(text)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.base_model.device)
        
        # Get True/False token ids
        true_tokens = tokenizer(" True", add_special_tokens=False)
        false_tokens = tokenizer(" False", add_special_tokens=False)
        true_token_id = true_tokens["input_ids"][0]
        false_token_id = false_tokens["input_ids"][0]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get probabilities for True/False tokens
            true_false_logits = logits[:, [true_token_id, false_token_id]]
            probs = torch.softmax(true_false_logits, dim=-1)
            
            # True = safe (0), False = unsafe (1)
            unsafe_prob = probs[0, 1].item()
            is_safe = unsafe_prob < 0.5
            
        return is_safe, 1 - unsafe_prob
    
    def get_token_importance(self, text: str, tokenizer: AutoTokenizer) -> np.ndarray:
        """Get importance scores for each token (for MI visualization)"""
        # This will be implemented with integrated gradients
        # Placeholder for now
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        return np.random.rand(encoded.input_ids.shape[1])
    
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