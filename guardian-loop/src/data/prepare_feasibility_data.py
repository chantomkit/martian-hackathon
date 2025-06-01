#!/usr/bin/env python3
"""
Prepare datasets for evaluating model response feasibility/truthfulness
Uses TruthfulQA, DefAn, and HaluEval datasets to create training data
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv

def load_truthfulqa_data() -> List[Dict[str, Any]]:
    """Load and process TruthfulQA dataset"""
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "multiple_choice")
    
    processed_data = []
    for item in tqdm(dataset['train']):
        # Get correct and incorrect answers
        correct_answers = item.get('correct_answers', [])
        incorrect_answers = item.get('incorrect_answers', [])
        
        # Use the first correct answer as the "model response" to evaluate
        if correct_answers:
            # Add a truthful example
            processed_data.append({
                'model_response': correct_answers[0],
                'question': item['question'],
                'is_truthful': True,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers,
                'source': 'truthfulqa'
            })
        
        # Use some incorrect answers as untruthful examples
        if incorrect_answers:
            # Add an untruthful example
            processed_data.append({
                'model_response': incorrect_answers[0],
                'question': item['question'],
                'is_truthful': False,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers,
                'source': 'truthfulqa'
            })
    
    return processed_data

def load_halueval_data() -> List[Dict[str, Any]]:
    """Load and process alternative hallucination dataset"""
    print("Loading alternative dataset for hallucination detection (vectara/hallucination_dataset)...")
    dataset = load_dataset("vectara/hallucination_dataset", split="train")
    
    processed_data = []
    for item in tqdm(dataset):
        response = item.get('response', '')
        hallucination = item.get('hallucination', True)  # Default to True if not specified
        
        if response:
            processed_data.append({
                'model_response': response,
                'question': item.get('prompt', 'Hallucination check'),
                'is_truthful': not hallucination,  # Invert hallucination flag
                'source': 'hallucination_dataset',
                'hallucination_type': item.get('type', 'unknown')
            })
    
    return processed_data

def prepare_feasibility_data(base_dir: str = './data'):
    """Main function to prepare feasibility datasets"""
    # Create output directory structure
    output_dir = Path(base_dir) / 'feasibility'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HuggingFace token if available
    load_dotenv()
    if 'HF_TOKEN' in os.environ:
        login(token=os.getenv('HF_TOKEN'))
    
    # Load all datasets
    all_data = []
    
    try:
        truthfulqa_data = load_truthfulqa_data()
        all_data.extend(truthfulqa_data)
        print(f"Loaded {len(truthfulqa_data)} samples from TruthfulQA")
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
    
    try:
        halueval_data = load_halueval_data()
        all_data.extend(halueval_data)
        print(f"Loaded {len(halueval_data)} samples from HaluEval")
    except Exception as e:
        print(f"Error loading HaluEval: {e}")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_data)
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Truthful responses: {df['is_truthful'].sum()}")
    print(f"Non-truthful responses: {len(df) - df['is_truthful'].sum()}")
    print("\nSamples per source:")
    print(df['source'].value_counts())
    
    # Balance dataset if needed
    min_samples = min(
        len(df[df['is_truthful']]),
        len(df[~df['is_truthful']])
    )
    
    truthful = df[df['is_truthful']].sample(min_samples, random_state=42)
    non_truthful = df[~df['is_truthful']].sample(min_samples, random_state=42)
    
    balanced_df = pd.concat([truthful, non_truthful])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
    
    # Split into train/val/test
    train_frac = 0.8
    val_frac = 0.1
    test_frac = 0.1
    
    train_size = int(len(balanced_df) * train_frac)
    val_size = int(len(balanced_df) * val_frac)
    
    train_df = balanced_df[:train_size]
    val_df = balanced_df[train_size:train_size + val_size]
    test_df = balanced_df[train_size + val_size:]
    
    # Convert to final format
    def convert_to_final_format(row):
        return {
            'text': row['model_response'],  # Model response is what we're evaluating
            'label': 1 if row['is_truthful'] else 0,  # 1 = truthful, 0 = not truthful
            'metadata': {
                'source': row['source'],
                'original_question': row['question'],
                'dataset_specific': {
                    k: v for k, v in row.items() 
                    if k not in ['model_response', 'is_truthful', 'source', 'question']
                }
            }
        }
    
    train_data = train_df.apply(convert_to_final_format, axis=1).tolist()
    val_data = val_df.apply(convert_to_final_format, axis=1).tolist()
    test_data = test_df.apply(convert_to_final_format, axis=1).tolist()
    
    # Save datasets with new structure
    for split_name, data in [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]:
        output_file = output_dir / f'{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print("\nSaved datasets:")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    print(f"Files saved to: {output_dir}")

if __name__ == '__main__':
    prepare_feasibility_data()  # Will save to ./data/feasibility/ 