"""
Safety Dataset Preparation for Guardian-Loop
Uses state-of-the-art safety datasets for training
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyDatasetPreparer:
    """Prepare safety datasets using specialized safety classification datasets"""
    
    def __init__(self):
        self.datasets = {}
        
    def load_wildguard(self, max_samples: int = 30000) -> List[Dict]:
        """Load WildGuard dataset - state-of-the-art safety dataset with 92K examples"""
        logger.info("Loading WildGuard dataset...")
        try:
            # WildGuard has both prompts and responses with safety labels
            dataset = load_dataset("allenai/wildguard", "wildguardmix", split="train")
            
            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Processing WildGuard")):
                if i >= max_samples:
                    break
                    
                # WildGuard format: prompt, response, prompt_harm_label, response_harm_label
                prompt = item.get('prompt', '')
                
                # Create unsafe examples from harmful prompts
                if item.get('prompt_harm_label', 0) == 1:  # Harmful prompt
                    samples.append({
                        'prompt': prompt,
                        'label': 0,  # Unsafe
                        'source': 'wildguard',
                        'categories': item.get('harm_categories', [])
                    })
                    
                # Create safe examples from benign prompts
                else:
                    samples.append({
                        'prompt': prompt,
                        'label': 1,  # Safe
                        'source': 'wildguard',
                        'categories': []
                    })
                    
            logger.info(f"Loaded {len(samples)} samples from WildGuard")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load WildGuard: {e}")
            return []
    
    def load_beavertails(self, max_samples: int = 30000) -> List[Dict]:
        """Load BeaverTails - 330K+ QA pairs with 14 harm categories"""
        logger.info("Loading BeaverTails dataset...")
        try:
            dataset = load_dataset("PKU-Alignment/BeaverTails", split="train")
            
            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Processing BeaverTails")):
                if i >= max_samples:
                    break
                    
                # BeaverTails has prompts with safety labels
                prompt = item.get('prompt', '')
                is_safe = item.get('is_safe', False)
                
                samples.append({
                    'prompt': prompt,
                    'label': 1 if is_safe else 0,
                    'source': 'beavertails',
                    'categories': item.get('category', [])
                })
                    
            logger.info(f"Loaded {len(samples)} samples from BeaverTails")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load BeaverTails: {e}")
            return []
    
    def load_toxicchat(self, max_samples: int = 10000) -> List[Dict]:
        """Load ToxicChat - real user-AI conversations with toxicity labels"""
        logger.info("Loading ToxicChat dataset...")
        try:
            dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
            
            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Processing ToxicChat")):
                if i >= max_samples:
                    break
                    
                # Use the human message as the prompt
                prompt = item.get('human', '') or item.get('user_input', '')
                # Check if it's toxic based on the annotation
                is_toxic = item.get('toxicity', 0) > 0 or item.get('jailbreak', 0) > 0
                
                samples.append({
                    'prompt': prompt,
                    'label': 0 if is_toxic else 1,  # 0 = unsafe, 1 = safe
                    'source': 'toxicchat',
                    'toxicity_score': item.get('toxicity', 0)
                })
                    
            logger.info(f"Loaded {len(samples)} samples from ToxicChat")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load ToxicChat: {e}")
            return []
    
    def load_wildchat_toxic(self, max_samples: int = 20000) -> List[Dict]:
        """Load toxic subset of WildChat - 150K+ real harmful conversations"""
        logger.info("Loading WildChat toxic subset...")
        try:
            # WildChat-1M has real conversations with toxicity annotations
            dataset = load_dataset("allenai/WildChat-1M", split="train")
            
            samples = []
            toxic_count = 0
            
            for item in tqdm(dataset, desc="Filtering WildChat for toxic content"):
                # Check if conversation contains toxic content
                if item.get('toxic', False) or item.get('redacted', False):
                    conversation = item.get('conversation', [])
                    if conversation and len(conversation) > 0:
                        # Extract user prompts that led to toxic content
                        prompt = conversation[0].get('content', '')
                        
                        samples.append({
                            'prompt': prompt,
                            'label': 0,  # Unsafe (led to toxic response)
                            'source': 'wildchat',
                            'model': item.get('model', 'unknown')
                        })
                        
                        toxic_count += 1
                        if toxic_count >= max_samples:
                            break
                            
            logger.info(f"Loaded {len(samples)} toxic samples from WildChat")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load WildChat: {e}")
            return []
    
    def load_anthropic_red_team(self, max_samples: int = 10000) -> List[Dict]:
        """Load Anthropic's red team attempts dataset"""
        logger.info("Loading Anthropic red team dataset...")
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", "red-team-attempts", split="train")
            
            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Processing red team data")):
                if i >= max_samples:
                    break
                    
                # Extract the red team prompt (usually harmful)
                text = item.get('chosen', '') or item.get('text', '')
                if '\n\nHuman:' in text:
                    prompt = text.split('\n\nHuman:')[1].split('\n\nAssistant:')[0].strip()
                    
                    samples.append({
                        'prompt': prompt,
                        'label': 0,  # Red team attempts are unsafe
                        'source': 'anthropic_red_team'
                    })
                    
            logger.info(f"Loaded {len(samples)} samples from Anthropic red team")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load Anthropic red team data: {e}")
            return []
    
    def mutate_unsafe_to_safe(self, unsafe_prompts: List[str], num_mutations: int = 1) -> List[Dict]:
        """
        Mutate unsafe prompts to create safe variations
        This helps the model learn boundaries between safe/unsafe
        """
        logger.info(f"Mutating {len(unsafe_prompts)} unsafe prompts to safe variations...")
        
        mutations = []
        
        # Mutation strategies
        safe_mutations = [
            # Ask about the topic academically
            lambda p: f"What are the ethical considerations regarding: {p}",
            lambda p: f"Can you explain why the following would be harmful: {p}",
            lambda p: f"What are the legal implications of: {p}",
            lambda p: f"How can we prevent: {p}",
            lambda p: f"What are the societal impacts of: {p}",
            # Reframe as research/educational
            lambda p: f"For educational purposes, what are the dangers of: {p}",
            lambda p: f"In a research context, how would one study: {p}",
            lambda p: f"What academic literature exists on: {p}",
            # Ask for alternatives
            lambda p: f"What are safe and legal alternatives to: {p}",
            lambda p: f"How can someone get help instead of: {p}",
        ]
        
        for prompt in tqdm(unsafe_prompts[:1000], desc="Creating safe mutations"):  # Limit mutations
            # Apply random mutations
            selected_mutations = np.random.choice(safe_mutations, size=min(num_mutations, len(safe_mutations)), replace=False)
            
            for mutation_fn in selected_mutations:
                try:
                    safe_prompt = mutation_fn(prompt)
                    mutations.append({
                        'prompt': safe_prompt,
                        'label': 1,  # Safe
                        'source': 'mutation',
                        'original_unsafe': prompt
                    })
                except:
                    continue
                    
        logger.info(f"Created {len(mutations)} safe mutations from unsafe prompts")
        return mutations
    
    def balance_dataset(self, samples: List[Dict], target_ratio: float = 0.5) -> List[Dict]:
        """Balance dataset to have target_ratio of safe prompts"""
        safe_samples = [s for s in samples if s['label'] == 1]
        unsafe_samples = [s for s in samples if s['label'] == 0]
        
        logger.info(f"Original distribution - Safe: {len(safe_samples)}, Unsafe: {len(unsafe_samples)}")
        
        # Calculate how many samples we need
        if len(safe_samples) / len(samples) < target_ratio:
            # Need more safe samples
            target_safe = int(len(unsafe_samples) * target_ratio / (1 - target_ratio))
            if target_safe > len(safe_samples):
                # Oversample safe or undersample unsafe
                if len(safe_samples) * 2 < target_safe:
                    # Undersample unsafe instead
                    target_unsafe = int(len(safe_samples) * (1 - target_ratio) / target_ratio)
                    unsafe_samples = np.random.choice(unsafe_samples, size=target_unsafe, replace=False).tolist()
                else:
                    # Oversample safe
                    additional_safe = target_safe - len(safe_samples)
                    safe_samples += np.random.choice(safe_samples, size=additional_safe, replace=True).tolist()
        else:
            # Need more unsafe samples (less common)
            target_unsafe = int(len(safe_samples) * (1 - target_ratio) / target_ratio)
            if target_unsafe < len(unsafe_samples):
                unsafe_samples = np.random.choice(unsafe_samples, size=target_unsafe, replace=False).tolist()
                
        balanced = safe_samples + unsafe_samples
        np.random.shuffle(balanced)
        
        final_safe = sum(1 for s in balanced if s['label'] == 1)
        logger.info(f"Balanced distribution - Safe: {final_safe}, Unsafe: {len(balanced) - final_safe}")
        logger.info(f"Final ratio of safe prompts: {final_safe/len(balanced):.2%}")
        
        return balanced
    
    def prepare_dataset(
        self, 
        output_dir: str = "./data/prepared",
        test_split: float = 0.2,
        val_split: float = 0.1,
        balance_ratio: float = 0.5,
        use_mutations: bool = True
    ):
        """Prepare complete dataset with train/val/test splits"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all datasets
        all_samples = []
        
        # Load specialized safety datasets
        all_samples.extend(self.load_wildguard(max_samples=30000))
        all_samples.extend(self.load_beavertails(max_samples=30000))
        all_samples.extend(self.load_toxicchat(max_samples=10000))
        all_samples.extend(self.load_wildchat_toxic(max_samples=20000))
        all_samples.extend(self.load_anthropic_red_team(max_samples=10000))
        
        # Get unsafe prompts for mutation
        if use_mutations:
            unsafe_prompts = [s['prompt'] for s in all_samples if s['label'] == 0]
            mutations = self.mutate_unsafe_to_safe(unsafe_prompts, num_mutations=1)
            all_samples.extend(mutations)
        
        # Remove duplicates based on prompt
        seen_prompts = set()
        unique_samples = []
        for sample in all_samples:
            prompt_lower = sample['prompt'].lower().strip()
            if prompt_lower not in seen_prompts and len(prompt_lower) > 10:
                seen_prompts.add(prompt_lower)
                unique_samples.append(sample)
        
        logger.info(f"Total unique samples: {len(unique_samples)}")
        
        # Balance the dataset
        balanced_samples = self.balance_dataset(unique_samples, target_ratio=balance_ratio)
        
        # Shuffle before splitting
        np.random.shuffle(balanced_samples)
        
        # Create splits
        n_samples = len(balanced_samples)
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * val_split)
        
        test_data = balanced_samples[:test_size]
        val_data = balanced_samples[test_size:test_size + val_size]
        train_data = balanced_samples[test_size + val_size:]
        
        # Print statistics
        self._print_statistics(train_data, val_data, test_data)
        
        # Save datasets
        self._save_dataset(train_data, os.path.join(output_dir, "train.json"))
        self._save_dataset(val_data, os.path.join(output_dir, "val.json"))
        self._save_dataset(test_data, os.path.join(output_dir, "test.json"))
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _print_statistics(self, train_data, val_data, test_data):
        """Print dataset statistics"""
        
        def get_stats(data, name):
            safe_count = sum(1 for s in data if s['label'] == 1)
            unsafe_count = len(data) - safe_count
            sources = defaultdict(int)
            for s in data:
                sources[s.get('source', 'unknown')] += 1
                
            print(f"\n{name} Dataset:")
            print(f"  Total: {len(data)}")
            print(f"  Safe: {safe_count} ({safe_count/len(data)*100:.1f}%)")
            print(f"  Unsafe: {unsafe_count} ({unsafe_count/len(data)*100:.1f}%)")
            print(f"  Sources: {dict(sources)}")
            
        get_stats(train_data, "Train")
        get_stats(val_data, "Validation")
        get_stats(test_data, "Test")
        
    def _save_dataset(self, data: List[Dict], filepath: str):
        """Save dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} samples to {filepath}")


if __name__ == "__main__":
    preparer = SafetyDatasetPreparer()
    preparer.prepare_dataset(
        balance_ratio=0.5,  # 50% safe, 50% unsafe
        use_mutations=True  # Use your mutation idea!
    ) 