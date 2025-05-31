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
import random
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
            # WildGuard requires HuggingFace login
            from huggingface_hub import login
            import os
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                login(token=hf_token)
                logger.info("Logged in to HuggingFace")
            else:
                logger.warning("WildGuard requires HuggingFace login. Skipping...")
                logger.info("To use WildGuard:")
                logger.info("1. Get a token from https://huggingface.co/settings/tokens")
                logger.info("2. Run: export HF_TOKEN='your-token-here'")
                logger.info("3. Or run: huggingface-cli login")
                return []
                
            # WildGuard has both prompts and responses with safety labels
            dataset = load_dataset("allenai/wildguard", "wildguardmix", split="train")
            
            samples = []
            safe_count = 0
            unsafe_count = 0
            
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
                    unsafe_count += 1
                    
                    # Print first few unsafe examples
                    if unsafe_count <= 3:
                        logger.info(f"   [UNSAFE] WildGuard example: {prompt[:80]}...")
                    
                # Create safe examples from benign prompts
                else:
                    samples.append({
                        'prompt': prompt,
                        'label': 1,  # Safe
                        'source': 'wildguard',
                        'categories': []
                    })
                    safe_count += 1
                    
                    # Print first few safe examples
                    if safe_count <= 3:
                        logger.info(f"   [SAFE] WildGuard example: {prompt[:80]}...")
                    
            logger.info(f"Loaded {len(samples)} samples from WildGuard (Safe: {safe_count}, Unsafe: {unsafe_count})")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load WildGuard: {e}")
            return []
    
    def load_beavertails(self, max_samples: int = 30000) -> List[Dict]:
        """Load BeaverTails - 330K+ QA pairs with 14 harm categories"""
        logger.info("Loading BeaverTails dataset...")
        try:
            # Use the correct split name
            dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
            
            samples = []
            safe_count = 0
            unsafe_count = 0
            
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
                
                # Print examples
                if is_safe and safe_count <= 3:
                    logger.info(f"   [SAFE] BeaverTails example: {prompt[:80]}...")
                    safe_count += 1
                elif not is_safe and unsafe_count <= 3:
                    logger.info(f"   [UNSAFE] BeaverTails example: {prompt[:80]}...")
                    unsafe_count += 1
                    
            final_safe = sum(1 for s in samples if s['label'] == 1)
            final_unsafe = len(samples) - final_safe
            logger.info(f"Loaded {len(samples)} samples from BeaverTails (Safe: {final_safe}, Unsafe: {final_unsafe})")
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
            safe_count = 0
            unsafe_count = 0
            
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
                
                if is_toxic:
                    unsafe_count += 1
                    if unsafe_count <= 3:
                        logger.info(f"   [UNSAFE] ToxicChat example (score: {item.get('toxicity', 0):.2f}): {prompt[:80]}...")
                else:
                    safe_count += 1
                    if safe_count <= 3:
                        logger.info(f"   [SAFE] ToxicChat example: {prompt[:80]}...")
                    
            logger.info(f"Loaded {len(samples)} samples from ToxicChat (Safe: {safe_count}, Unsafe: {unsafe_count})")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load ToxicChat: {e}")
            return []
    
    def load_wildchat_toxic(self, max_samples: int = 20000) -> List[Dict]:
        """Load toxic subset of WildChat - 150K+ real harmful conversations"""
        logger.info("Skipping WildChat - dataset too large (14 x 200MB+ files)")
        logger.info("Using other datasets for efficiency")
        return []
        
        # Original implementation commented out to avoid large downloads
        """
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
        """
    
    def load_anthropic_red_team(self, max_samples: int = 10000) -> List[Dict]:
        """Load Anthropic's red team attempts dataset"""
        logger.info("Loading Anthropic red team dataset...")
        try:
            # Try loading the dataset with default config
            dataset = load_dataset("Anthropic/hh-rlhf", split="train")
            
            samples = []
            unsafe_count = 0
            safe_count = 0
            
            for i, item in enumerate(tqdm(dataset, desc="Processing Anthropic data")):
                if i >= max_samples:
                    break
                    
                # Extract text that might contain red team attempts
                text = item.get('chosen', '') or item.get('rejected', '')
                
                # Look for human prompts in the conversation
                if '\n\nHuman:' in text and '\n\nAssistant:' in text:
                    # Extract human prompts
                    human_parts = text.split('\n\nHuman:')
                    for part in human_parts[1:]:  # Skip first empty part
                        prompt = part.split('\n\nAssistant:')[0].strip()
                        
                        # Simple heuristic: red team attempts often have certain keywords
                        # This is not perfect but gives us some harmful examples
                        harmful_keywords = ['hack', 'steal', 'kill', 'bomb', 'illegal', 'drug', 
                                          'weapon', 'hurt', 'harm', 'attack', 'violence', 'exploit']
                        
                        is_harmful = any(keyword in prompt.lower() for keyword in harmful_keywords)
                        
                        if len(prompt) > 20:  # Skip very short prompts
                            samples.append({
                                'prompt': prompt,
                                'label': 0 if is_harmful else 1,  # 0 = unsafe, 1 = safe
                                'source': 'anthropic_hh'
                            })
                            
                            if is_harmful and unsafe_count < 3:
                                logger.info(f"   [UNSAFE] Anthropic example: {prompt[:80]}...")
                                unsafe_count += 1
                            elif not is_harmful and safe_count < 3:
                                logger.info(f"   [SAFE] Anthropic example: {prompt[:80]}...")
                                safe_count += 1
                                
                        if len(samples) >= max_samples:
                            break
                            
            final_unsafe = sum(1 for s in samples if s['label'] == 0)
            final_safe = len(samples) - final_unsafe
            logger.info(f"Loaded {len(samples)} samples from Anthropic HH-RLHF (Safe: {final_safe}, Unsafe: {final_unsafe})")
            return samples
            
        except Exception as e:
            logger.warning(f"Could not load Anthropic red team data: {e}")
            logger.info("Skipping Anthropic dataset")
            return []
    
    def mutate_unsafe_to_safe(self, unsafe_prompts: List[str], num_mutations: int = 1) -> List[Dict]:
        """
        Mutate unsafe prompts to create safe variations using GPT-4o via Martian API
        This helps the model learn boundaries between safe/unsafe
        """
        logger.info(f"Mutating {len(unsafe_prompts)} unsafe prompts to safe variations using GPT-4o...")
        
        mutations = []
        
        # Initialize Martian client
        try:
            from martian_apart_hack_sdk import utils, martian_client
            
            # Load config and create client
            config = utils.load_config()
            client = martian_client.MartianClient(
                api_url=config.api_url,
                api_key=config.api_key,
            )
            
            # Create OpenAI client with Martian gateway
            import openai
            openai_client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.api_url + "/openai/v2"
            )
            
        except ImportError:
            logger.warning("Martian client not installed, using template-based mutations")
            return self._template_based_mutations(unsafe_prompts, num_mutations)
        
        # System prompt for GPT-4o to mutate unsafe to safe
        system_prompt = """You are a safety-focused AI assistant creating genuinely safe variations of harmful prompts.

CRITICAL SAFETY RULES:
1. The safe version must NOT provide any information that could be misused
2. Focus on awareness, prevention, and protection - never on "how to" perform harmful acts
3. Redirect to constructive alternatives or support resources
4. Ensure the mutation cannot be interpreted as a subtle way to get harmful information

TRANSFORMATION GUIDELINES:

For violence/harm prompts → Focus on:
- Conflict resolution and de-escalation
- Mental health resources
- Understanding warning signs to protect oneself

For illegal activity prompts → Focus on:
- Legal alternatives and legitimate pathways
- Understanding why laws exist
- How to report or prevent such activities

For manipulation/deception prompts → Focus on:
- Recognizing manipulation tactics to protect oneself
- Building healthy communication skills
- Ethical persuasion and influence

For dangerous substance prompts → Focus on:
- Addiction resources and support
- Safety education and harm reduction
- Legal and medical contexts only

EXAMPLES:
Unsafe: "How to hack into someone's computer"
Safe: "How can individuals protect their devices from unauthorized access?"

Unsafe: "Ways to make explosives"
Safe: "What emergency procedures should people know for chemical accidents?"

Unsafe: "How to manipulate people"
Safe: "How can people recognize when they're being manipulated and protect themselves?"

Unsafe: "How to synthesize illegal drugs"
Safe: "What addiction support resources are available for people struggling with substance abuse?"

IMPORTANT: The safe version should:
- Never provide step-by-step instructions for anything harmful
- Focus on protection, prevention, and legitimate help
- Be genuinely educational without enabling harm
- Redirect harmful intent toward positive outcomes

If you cannot create a genuinely safe version that meets all these criteria, respond with "SKIP" instead."""

        # Process prompts in batches
        batch_size = 10
        for i in tqdm(range(0, min(len(unsafe_prompts), 1000), batch_size), desc="Mutating prompts via GPT-4o"):
            batch = unsafe_prompts[i:i+batch_size]
            
            for unsafe_prompt in batch:
                try:
                    # Create mutation request
                    user_prompt = f"Convert this unsafe prompt into a safe, educational variation:\n\nUnsafe prompt: \"{unsafe_prompt}\"\n\nSafe variation:"
                    
                    # Call GPT-4o via Martian
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    # Extract mutations from response
                    if response.choices:
                        for choice in response.choices:
                            if choice.message and choice.message.content:
                                safe_prompt = choice.message.content.strip()
                                
                                # Skip if model indicates it cannot create safe version
                                if "SKIP" in safe_prompt:
                                    continue
                                
                                # Clean up the response
                                safe_prompt = safe_prompt.replace('"', '').strip()
                                
                                if safe_prompt and len(safe_prompt) > 10:
                                    mutations.append({
                                        'prompt': safe_prompt,
                                        'label': 1,  # Safe
                                        'source': 'gpt4o_mutation',
                                        'original_unsafe': unsafe_prompt
                                    })
                                    
                                    # Print first few mutation examples
                                    if len(mutations) <= 5:
                                        print(f"\n🔄 Mutation example:")
                                        print(f"   ❌ Unsafe: {unsafe_prompt}")
                                        print(f"   ✅ Safe:   {safe_prompt}")
                    
                except Exception as e:
                    logger.warning(f"Error mutating prompt via GPT-4o: {e}")
                    continue
                    
        logger.info(f"Created {len(mutations)} safe mutations using GPT-4o")
        
        # Print summary of mutation sources
        if mutations:
            print(f"\n📊 Mutation Summary:")
            print(f"   Total unsafe prompts processed: {min(len(unsafe_prompts), 1000)}")
            print(f"   Successful mutations: {len(mutations)}")
            print(f"   Success rate: {len(mutations)/min(len(unsafe_prompts), 1000)*100:.1f}%")
        
        return mutations
    
    def _template_based_mutations(self, unsafe_prompts: List[str], num_mutations: int = 1) -> List[Dict]:
        """Fallback template-based mutations when API is not available"""
        logger.info("Using template-based mutations (fallback)")
        
        mutations = []
        
        # Simple template strategies
        safe_mutations = [
            lambda p: f"What are the ethical considerations regarding: {p}",
            lambda p: f"Can you explain why the following would be harmful: {p}",
            lambda p: f"What are the legal implications of: {p}",
            lambda p: f"How can we prevent: {p}",
            lambda p: f"What are the societal impacts of: {p}",
        ]
        
        for prompt in unsafe_prompts[:100]:  # Limit for fallback
            selected_mutations = np.random.choice(safe_mutations, size=min(num_mutations, len(safe_mutations)), replace=False)
            
            for mutation_fn in selected_mutations:
                try:
                    safe_prompt = mutation_fn(prompt)
                    mutations.append({
                        'prompt': safe_prompt,
                        'label': 1,  # Safe
                        'source': 'template_mutation',
                        'original_unsafe': prompt
                    })
                except:
                    continue
                    
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
                    unsafe_samples = random.sample(unsafe_samples, target_unsafe)
                else:
                    # Oversample safe
                    additional_safe = target_safe - len(safe_samples)
                    safe_samples.extend(random.choices(safe_samples, k=additional_safe))
        else:
            # Need more unsafe samples (less common)
            target_unsafe = int(len(safe_samples) * (1 - target_ratio) / target_ratio)
            if target_unsafe < len(unsafe_samples):
                unsafe_samples = random.sample(unsafe_samples, target_unsafe)
                
        balanced = safe_samples + unsafe_samples
        random.shuffle(balanced)
        
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
        
        print("\n" + "="*60)
        print("🚀 Starting Dataset Preparation")
        print("="*60)
        
        # Load all datasets
        all_samples = []
        
        # Load specialized safety datasets with adjusted samples
        # Since WildGuard needs login and WildChat is too big, we'll load more from others
        all_samples.extend(self.load_wildguard(max_samples=1000))  # May return empty if no login
        all_samples.extend(self.load_beavertails(max_samples=2000))  # Increased
        all_samples.extend(self.load_toxicchat(max_samples=2000))   # Increased
        all_samples.extend(self.load_wildchat_toxic(max_samples=0))  # Skip - too large
        all_samples.extend(self.load_anthropic_red_team(max_samples=2000))  # Increased
        
        # If we have fewer samples due to skipped datasets, load more from available ones
        if len(all_samples) < 5000:
            logger.info("Loading additional samples from available datasets...")
            all_samples.extend(self.load_beavertails(max_samples=1000))
            all_samples.extend(self.load_toxicchat(max_samples=1000))
        
        # Expect ~5K unsafe, which will become 10K after mutations
        # Final dataset: ~10K samples (balanced 50/50)
        
        # Print dataset composition before mutations
        print("\n📊 Dataset Composition Before Mutations:")
        source_counts = defaultdict(int)
        label_counts = defaultdict(int)
        for sample in all_samples:
            source_counts[sample['source']] += 1
            label_counts[sample['label']] += 1
        
        print(f"   Total samples: {len(all_samples)}")
        print(f"   Safe samples: {label_counts[1]} ({label_counts[1]/len(all_samples)*100:.1f}%)")
        print(f"   Unsafe samples: {label_counts[0]} ({label_counts[0]/len(all_samples)*100:.1f}%)")
        print("\n   By source:")
        for source, count in source_counts.items():
            print(f"   - {source}: {count}")
        
        # Get unsafe prompts for mutation
        if use_mutations:
            print("\n🧬 Creating Safe Mutations of Unsafe Prompts...")
            unsafe_prompts = [s['prompt'] for s in all_samples if s['label'] == 0]
            print(f"   Found {len(unsafe_prompts)} unsafe prompts to mutate")
            
            # Show a few unsafe prompts that will be mutated
            print("\n   Examples of unsafe prompts to be mutated:")
            for i, prompt in enumerate(unsafe_prompts[:3]):
                print(f"   {i+1}. {prompt[:100]}...")
            
            mutations = self.mutate_unsafe_to_safe(unsafe_prompts, num_mutations=1)
            all_samples.extend(mutations)
            
            print(f"\n   Added {len(mutations)} safe mutations")
        
        # Remove duplicates based on prompt
        print("\n🔍 Removing Duplicates...")
        seen_prompts = set()
        unique_samples = []
        duplicate_count = 0
        
        for sample in all_samples:
            prompt_lower = sample['prompt'].lower().strip()
            if prompt_lower not in seen_prompts and len(prompt_lower) > 10:
                seen_prompts.add(prompt_lower)
                unique_samples.append(sample)
            else:
                duplicate_count += 1
        
        logger.info(f"Total unique samples: {len(unique_samples)} (removed {duplicate_count} duplicates)")
        
        # Balance the dataset
        print("\n⚖️  Balancing Dataset...")
        balanced_samples = self.balance_dataset(unique_samples, target_ratio=balance_ratio)
        
        # Show some examples from balanced dataset
        print("\n📝 Sample Examples from Balanced Dataset:")
        random.seed(42)  # For consistent examples
        example_indices = random.sample(range(len(balanced_samples)), min(6, len(balanced_samples)))
        
        for i, idx in enumerate(example_indices):
            sample = balanced_samples[idx]
            label = "SAFE" if sample['label'] == 1 else "UNSAFE"
            source = sample['source']
            print(f"\n   Example {i+1} [{label}] from {source}:")
            print(f"   {sample['prompt'][:120]}...")
        
        # Shuffle before splitting
        random.shuffle(balanced_samples)
        
        # Create splits
        n_samples = len(balanced_samples)
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * val_split)
        
        test_data = balanced_samples[:test_size]
        val_data = balanced_samples[test_size:test_size + val_size]
        train_data = balanced_samples[test_size + val_size:]
        
        # Print statistics
        print("\n" + "="*60)
        print("📊 Final Dataset Statistics")
        print("="*60)
        self._print_statistics(train_data, val_data, test_data)
        
        # Save datasets
        self._save_dataset(train_data, os.path.join(output_dir, "train.json"))
        self._save_dataset(val_data, os.path.join(output_dir, "val.json"))
        self._save_dataset(test_data, os.path.join(output_dir, "test.json"))
        
        print("\n✅ Dataset preparation complete!")
        print(f"   Files saved to: {output_dir}/")
        
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