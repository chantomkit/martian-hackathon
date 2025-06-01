#!/usr/bin/env python3
"""
Prepare datasets for evaluating model response feasibility/truthfulness
Uses TruthfulQA, DefAn, and HaluEval datasets to create training data
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv
import openai
from martian_apart_hack_sdk import utils, martian_client

async def get_gpt4o_evaluation(client: openai.OpenAI, question: str, ground_truth: str, llm_answer: str) -> Tuple[bool, str]:
    """Use GPT-4o to evaluate if the LLM's answer is equivalent to the ground truth"""
    try:
        system_prompt = """You are an expert evaluator determining if two answers to a question are semantically equivalent.
Compare the ground truth answer with the LLM's answer and determine if they convey the same meaning and information.
Consider:
1. Core information accuracy
2. Semantic equivalence (even if worded differently)
3. Completeness of information
4. Logical consistency

Output format:
{
    "is_equivalent": true/false,
    "explanation": "Brief explanation of your reasoning"
}

Be strict in your evaluation - answers should convey the same core meaning to be considered equivalent."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Question: {question}

Ground Truth Answer: {ground_truth}

LLM Answer: {llm_answer}

Are these answers semantically equivalent? Provide your evaluation in the specified JSON format."""}
            ],
            temperature=0.1,
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        return bool(result["is_equivalent"]), str(result["explanation"])
    except Exception as e:
        print(f"Error in GPT-4 evaluation: {str(e)}")
        return False, f"Error during evaluation: {str(e)}"

async def get_llama_response(client: openai.OpenAI, prompt: str) -> str:
    """Get response from LLAMA 3.3 70B Instruct Turbo for a given prompt"""
    try:
        response = client.chat.completions.create(
            model="together/meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Correct model name
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Please provide a direct and concise answer to the user's question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting LLAMA response for prompt: {prompt[:50]}...")
        print(f"Error: {str(e)}")
        return ""

def load_truthfulqa_data() -> List[Dict[str, Any]]:
    """Load and process TruthfulQA dataset, keeping only truthful examples"""
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation")
    
    processed_data = []
    for item in tqdm(dataset['validation']):
        # Get correct answers only
        correct_answers = item.get('correct_answers', [])
        
        # Only use the first correct answer as a truthful example
        if correct_answers:
            processed_data.append({
                'model_response': correct_answers[0],
                'question': item['question'],
                'is_truthful': True,
                'correct_answers': correct_answers,
                'source': 'truthfulqa'
            })
    
    return processed_data

def load_halueval_data() -> List[Dict[str, Any]]:
    """Load and process hallucination dataset, keeping only non-hallucinated examples"""
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
            # In HaluEval, hallucination=yes means the answer is not truthful, no means it is truthful
            hallucination = bool(item.get('hallucination') == 'yes')
            
            # Only keep non-hallucinated (truthful) examples
            if answer and not hallucination:
                processed_data.append({
                    'model_response': answer,
                    'question': question,
                    'is_truthful': True,
                    'source': 'halueval',
                    'context': knowledge  # Store knowledge as context for reference
                })
    
    print(f"\nProcessed {len(data)} total examples")
    print(f"Found {len(processed_data)} non-hallucinated examples")
    return processed_data

async def convert_to_final_format_with_gpt(row, client: openai.OpenAI):
    """Convert row to final format with LLAMA response and GPT-4o evaluation"""
    # Get LLAMA-3.3-70B response
    llama_response = await get_llama_response(client, row['question'])
    
    # Get GPT-4o's evaluation of answer equivalence
    is_equivalent, explanation = await get_gpt4o_evaluation(
        client,
        row['question'],
        row['model_response'],  # This is our ground truth
        llama_response
    )
    
    return {
        'prompt': row['question'],
        'answer': row['model_response'],  # Ground truth answer
        'llm_answer': llama_response,  # LLAMA's attempt
        'label': 1 if is_equivalent else 0,  # 1 if GPT-4o considers them equivalent
        'metadata': {
            'source': row['source'],
            'dataset_specific': {
                k: v for k, v in row.items() 
                if k not in ['model_response', 'is_truthful', 'source', 'question']
            },
            'evaluation': {
                'explanation': explanation,
                'model': 'gpt-4o',
                'llm_model': 'Llama-3.3-70B-Instruct-Turbo'
            }
        }
    }

async def process_dataframe(df: pd.DataFrame, client: openai.OpenAI):
    """Process DataFrame with concurrent API calls"""
    tasks = []
    for _, row in df.iterrows():
        tasks.append(convert_to_final_format_with_gpt(row, client))
    return await asyncio.gather(*tasks)

def prepare_feasibility_data(
    output_dir: str = "./data/feasibility",
    train_split: float = 0.8,
    total_dataset_size: int = 10
) -> None:
    """Main function to prepare feasibility datasets"""
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HuggingFace token if available
    load_dotenv()
    if 'HF_TOKEN' in os.environ:
        login(token=os.getenv('HF_TOKEN'))
    
    # Initialize Martian client
    config = utils.load_config()
    openai_client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.api_url + "/openai/v2"
    )
    
    # Load all datasets
    all_data = []
    
    try:
        truthfulqa_data = load_truthfulqa_data()
        all_data.extend(truthfulqa_data)
        print(f"Loaded {len(truthfulqa_data)} truthful samples from TruthfulQA")
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
    
    try:
        halueval_data = load_halueval_data()
        all_data.extend(halueval_data)
        print(f"Loaded {len(halueval_data)} truthful samples from HaluEval")
    except Exception as e:
        print(f"Error loading HaluEval: {e}")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_data)
    
    # Basic statistics before sampling
    print("\nInitial Dataset Statistics:")
    print(f"Total truthful samples: {len(df)}")
    print("\nSamples per source:")
    print(df['source'].value_counts())
    
    # Sample data BEFORE splitting to reduce API calls
    if len(df) > total_dataset_size:
        print(f"\nSampling {total_dataset_size} examples from {len(df)} total samples...")
        df = df.sample(total_dataset_size, random_state=42)
    
    # Split into train/val/test
    val_test_frac = (1 - train_split) / 2
    
    train_size = int(len(df) * train_split)
    val_size = int(len(df) * val_test_frac)
    
    # Ensure at least one sample in each split
    train_size = max(1, train_size)
    val_size = max(1, val_size)
    test_size = max(1, len(df) - train_size - val_size)
    
    print(f"\nSplit sizes:")
    print(f"Train: {train_size}")
    print(f"Val: {val_size}")
    print(f"Test: {test_size}")
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:train_size + val_size + test_size]
    
    # Process each split
    print("\nGetting model responses and evaluations...")
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        
        print(f"\nProcessing train split ({len(train_df)} samples)...")
        train_data = loop.run_until_complete(process_dataframe(train_df, openai_client))
        
        print(f"Processing validation split ({len(val_df)} samples)...")
        val_data = loop.run_until_complete(process_dataframe(val_df, openai_client))
        
        print(f"Processing test split ({len(test_df)} samples)...")
        test_data = loop.run_until_complete(process_dataframe(test_df, openai_client))
    
    # Calculate and print evaluation statistics
    def get_stats(data):
        total = len(data)
        feasible = sum(1 for item in data if item['label'] == 1)
        return {
            'total': total,
            'feasible': feasible,
            'feasible_rate': feasible / total if total > 0 else 0
        }
    
    print("\nFinal Dataset Statistics:")
    for split_name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        stats = get_stats(data)
        print(f"\n{split_name} Split:")
        print(f"Total samples: {stats['total']}")
        print(f"Feasible responses: {stats['feasible']} ({stats['feasible_rate']:.1%})")
    
    # Save datasets with new structure
    for split_name, data in [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]:
        output_file = output_dir / f'{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"\nFiles saved to: {output_dir}")

if __name__ == '__main__':
    # Example usage with default parameters
    prepare_feasibility_data() 