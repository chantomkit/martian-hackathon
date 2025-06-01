#!/usr/bin/env python3
"""
Inference script for the Safety Judge model
Allows testing individual prompts
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import argparse
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.safety_judge import SafetyJudge, SafetyJudgeConfig


def load_model(checkpoint_path: Path):
    """Load the trained safety judge model"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with saved config
    model_config = checkpoint.get('model_config')
    if model_config is None:
        # Use default config if not saved
        model_config = SafetyJudgeConfig()
    
    model = SafetyJudge(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, model_config, device


def run_inference(prompt: str, model, tokenizer, model_config, device):
    """Run inference on a single prompt"""
    # Format prompt
    formatted_prompt = model_config.prompt_template.format(text=prompt)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model_config.max_length
    ).to(device)
    
    # Get True/False token ids
    true_tokens = tokenizer.encode(" True", add_special_tokens=False)
    false_tokens = tokenizer.encode(" False", add_special_tokens=False)
    true_token_id = true_tokens[0]
    false_token_id = false_tokens[0]
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
        
        # Get logits for True/False tokens
        true_false_logits = logits[:, -1, [true_token_id, false_token_id]]
        probs = torch.softmax(true_false_logits, dim=-1)
        
        # True = safe (1), False = unsafe (0)
        is_safe = probs[0, 0] > probs[0, 1]
        confidence = probs[0, 0].item() if is_safe else probs[0, 1].item()
    
    return is_safe, confidence


def main():
    parser = argparse.ArgumentParser(description="Run inference with Safety Judge")
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt to analyze')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model, model_config, device = load_model(model_path)
    
    # Run inference
    is_safe, confidence = run_inference(args.prompt, model, tokenizer, model_config, device)
    
    # Output results
    if args.json:
        result = {
            'prompt': args.prompt,
            'is_safe': is_safe,
            'confidence': confidence,
            'classification': 'safe' if is_safe else 'unsafe'
        }
        print(json.dumps(result, indent=2))
    else:
        classification = "Safe" if is_safe else "Unsafe"
        print(f"{classification} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main() 