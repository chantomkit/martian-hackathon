#!/usr/bin/env python3
"""
Prepare datasets for evaluating model response feasibility/truthfulness
Uses TruthfulQA, DefAn, and HaluEval datasets to create training data
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv

def load_truthfulqa_data() -> List[Dict[str, Any]]:
    """Load and process TruthfulQA dataset"""
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation")
    
    processed_data = []
    for item in tqdm(dataset['validation']):
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
    """Load and process hallucination dataset"""
    print("Loading HaluEval dataset...")
    dataset: DatasetDict = load_dataset("pminervini/HaluEval", 'qa_samples')
    
    processed_data = []
    if isinstance(dataset, DatasetDict) and 'data' in dataset:
        data: Dataset = dataset['data']
        for item in tqdm(data):
            # Access fields directly since Dataset provides dictionary-like access
            knowledge = str(item['knowledge']) if 'knowledge' in item else ''
            question = str(item['question']) if 'question' in item else ''
            answer = str(item['answer']) if 'answer' in item else ''
            hallucination = bool(item['hallucination']) if 'hallucination' in item else True
            
            if answer:
                processed_data.append({
                    'model_response': answer,
                    'question': question,
                    'is_truthful': not hallucination,  # Convert hallucination flag to truthfulness
                    'source': 'halueval',
                    'context': knowledge  # Store knowledge as context for reference
                })
    
    return processed_data

def prepare_feasibility_data(
    output_dir: str = "./data/feasibility",
    train_split: float = 0.8,
    total_dataset_size: int = 10000,
    balance_ratio: float = 0.5
) -> None:
    """Main function to prepare feasibility datasets
    
    Args:
        output_dir: Directory to save the prepared datasets
        train_split: Fraction of data to use for training (remaining split between val/test)
        total_dataset_size: Target total size of the dataset
        balance_ratio: Target ratio of truthful to untruthful examples
    """
    # Create output directory structure
    output_dir = Path(output_dir)
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
    
    # Basic statistics before balancing
    print("\nInitial Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Truthful responses: {df['is_truthful'].sum()}")
    print(f"Non-truthful responses: {len(df) - df['is_truthful'].sum()}")
    print("\nSamples per source:")
    print(df['source'].value_counts())
    
    # Balance dataset according to balance_ratio
    truthful = df[df['is_truthful']]
    non_truthful = df[~df['is_truthful']]
    
    # Calculate target sizes based on total_dataset_size and balance_ratio
    target_truthful = int(total_dataset_size * balance_ratio)
    target_non_truthful = total_dataset_size - target_truthful
    
    # Sample or upsample to reach target sizes
    if len(truthful) > target_truthful:
        truthful = truthful.sample(target_truthful, random_state=42)
    else:
        truthful = truthful.sample(target_truthful, replace=True, random_state=42)
        
    if len(non_truthful) > target_non_truthful:
        non_truthful = non_truthful.sample(target_non_truthful, random_state=42)
    else:
        non_truthful = non_truthful.sample(target_non_truthful, replace=True, random_state=42)
    
    balanced_df = pd.concat([truthful, non_truthful])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
    
    # Split into train/val/test
    val_test_frac = (1 - train_split) / 2
    
    train_size = int(len(balanced_df) * train_split)
    val_size = int(len(balanced_df) * val_test_frac)
    
    train_df = balanced_df[:train_size]
    val_df = balanced_df[train_size:train_size + val_size]
    test_df = balanced_df[train_size + val_size:]
    
    # Convert to final format
    def convert_to_final_format(row):
        return {
            'prompt': row['question'],  # Store the original question/prompt
            'answer': row['model_response'],  # Store the model's response
            'label': 1 if row['is_truthful'] else 0,  # 1 = truthful, 0 = not truthful
            'metadata': {
                'source': row['source'],
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
    
    print("\nFinal Dataset Statistics:")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    print(f"Files saved to: {output_dir}")

if __name__ == '__main__':
    # Example usage with default parameters
    prepare_feasibility_data() 