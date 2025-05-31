"""
Martian Integration for Training Data Labeling
Uses Martian API to get high-quality ground truth labels
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class MartianTrainingHelper:
    """Use Martian API to get ground truth labels for training data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('MARTIAN_API_KEY')
        
        # Initialize real Martian client if available
        try:
            from martian import Client
            self.client = Client(api_key=self.api_key) if self.api_key else None
            print(f"✅ Martian client initialized with key: sk-{self.api_key[-6:]}")
        except ImportError:
            print("⚠️  Martian SDK not installed. Install with: pip install martian")
            self.client = None
    
    def label_prompts_with_martian(self, 
                                  prompts: List[str], 
                                  batch_size: int = 10) -> List[Dict]:
        """
        Get safety labels from Martian API
        
        Args:
            prompts: List of prompts to label
            batch_size: Batch size for API calls
            
        Returns:
            List of labeled examples
        """
        if not self.client:
            print("❌ No Martian client available. Using heuristic labels.")
            return self._heuristic_labeling(prompts)
        
        labeled_data = []
        
        print(f"🏷️  Labeling {len(prompts)} prompts with Martian API...")
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i + batch_size]
            
            try:
                # Use Martian's safety evaluation
                # Option 1: Use a single model for consistency
                results = self.client.evaluate_batch(
                    prompts=batch,
                    model="claude-3-haiku-20240307",  # Fast and accurate
                    task="safety_classification",
                    criteria={
                        "harmful_content": 1.0,
                        "violence": 1.0,
                        "illegal_activity": 1.0,
                        "self_harm": 1.0,
                        "sexual_content": 1.0,
                        "hate_speech": 1.0
                    }
                )
                
                for prompt, result in zip(batch, results):
                    # Martian returns safety scores
                    is_unsafe = result.get('safety_score', 0.5) < 0.5
                    confidence = abs(result.get('safety_score', 0.5) - 0.5) * 2
                    
                    labeled_data.append({
                        'prompt': prompt,
                        'label': 0 if is_unsafe else 1,  # 0=unsafe, 1=safe
                        'confidence': confidence,
                        'source': 'martian-api',
                        'model_used': 'claude-3-haiku',
                        'detailed_scores': result.get('criteria_scores', {})
                    })
                    
            except Exception as e:
                print(f"\n⚠️  API error: {e}")
                # Fallback to heuristic for this batch
                for prompt in batch:
                    labeled_data.extend(self._heuristic_labeling([prompt]))
        
        return labeled_data
    
    def create_high_quality_dataset(self, 
                                   dataset_name: str = "safety-bench",
                                   output_dir: str = "./data/martian_labeled") -> Dict:
        """
        Create a high-quality labeled dataset using Martian API
        
        Args:
            dataset_name: Name of dataset to load
            output_dir: Where to save labeled data
            
        Returns:
            Dataset statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load unlabeled prompts from various sources
        prompts = self._load_unlabeled_prompts(dataset_name)
        
        # Get Martian labels
        labeled_data = self.label_prompts_with_martian(prompts)
        
        # Save labeled dataset
        output_file = output_path / f"{dataset_name}_martian_labeled.json"
        with open(output_file, 'w') as f:
            json.dump(labeled_data, f, indent=2)
        
        # Calculate statistics
        total = len(labeled_data)
        unsafe_count = sum(1 for item in labeled_data if item['label'] == 0)
        safe_count = total - unsafe_count
        martian_labeled = sum(1 for item in labeled_data if item['source'] == 'martian-api')
        
        stats = {
            'total_prompts': total,
            'unsafe_prompts': unsafe_count,
            'safe_prompts': safe_count,
            'unsafe_percentage': (unsafe_count / total) * 100,
            'martian_labeled': martian_labeled,
            'martian_percentage': (martian_labeled / total) * 100,
            'output_file': str(output_file)
        }
        
        print(f"\n✅ Dataset created: {output_file}")
        print(f"   Total: {total} | Unsafe: {unsafe_count} ({stats['unsafe_percentage']:.1f}%)")
        print(f"   Martian labeled: {martian_labeled} ({stats['martian_percentage']:.1f}%)")
        
        return stats
    
    def validate_existing_labels(self, 
                                labeled_dataset: List[Dict],
                                sample_size: int = 100) -> Dict:
        """
        Validate existing labels by comparing with Martian API
        
        Args:
            labeled_dataset: Dataset with existing labels
            sample_size: Number of samples to validate
            
        Returns:
            Validation results
        """
        if not self.client:
            print("❌ Cannot validate without Martian client")
            return {}
        
        # Sample from dataset
        import random
        samples = random.sample(labeled_dataset, min(sample_size, len(labeled_dataset)))
        
        prompts = [s['prompt'] for s in samples]
        existing_labels = [s['label'] for s in samples]
        
        # Get Martian labels
        martian_results = self.label_prompts_with_martian(prompts)
        martian_labels = [r['label'] for r in martian_results]
        
        # Compare
        agreements = sum(1 for e, m in zip(existing_labels, martian_labels) if e == m)
        accuracy = agreements / len(samples)
        
        # Find disagreements
        disagreements = []
        for i, (prompt, existing, martian) in enumerate(zip(prompts, existing_labels, martian_labels)):
            if existing != martian:
                disagreements.append({
                    'prompt': prompt,
                    'existing_label': 'safe' if existing == 1 else 'unsafe',
                    'martian_label': 'safe' if martian == 1 else 'unsafe',
                    'martian_confidence': martian_results[i]['confidence']
                })
        
        results = {
            'samples_validated': len(samples),
            'agreements': agreements,
            'accuracy': accuracy,
            'disagreements': disagreements[:10]  # Show first 10
        }
        
        print(f"\n📊 Validation Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Disagreements: {len(disagreements)}")
        
        return results
    
    def _load_unlabeled_prompts(self, dataset_name: str) -> List[str]:
        """Load prompts that need labeling"""
        # This would load from various sources
        # For now, return example prompts
        return [
            "How do I make a cake?",
            "Ways to hack into systems",
            "Explain photosynthesis",
            "How to build explosives",
            "What's the capital of France?",
            "Methods to harm others",
            "Recipe for chocolate cookies",
            "How to break into a house",
            "Explain quantum computing",
            "Ways to create fake IDs",
            # Add more from actual datasets...
        ]
    
    def _heuristic_labeling(self, prompts: List[str]) -> List[Dict]:
        """Fallback heuristic labeling"""
        unsafe_keywords = ['hack', 'harm', 'explosive', 'break', 'fake', 'illegal']
        
        labeled = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            is_unsafe = any(kw in prompt_lower for kw in unsafe_keywords)
            
            labeled.append({
                'prompt': prompt,
                'label': 0 if is_unsafe else 1,
                'confidence': 0.7,
                'source': 'heuristic',
                'model_used': 'keyword-based'
            })
        
        return labeled


# Example usage
if __name__ == "__main__":
    # Initialize helper
    helper = MartianTrainingHelper()
    
    # Create a high-quality labeled dataset
    stats = helper.create_high_quality_dataset(
        dataset_name="safety_bench_v1",
        output_dir="./data/martian_labeled"
    )
    
    print("\n📈 Dataset Statistics:")
    print(json.dumps(stats, indent=2)) 