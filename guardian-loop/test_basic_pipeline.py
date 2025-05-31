"""
Test script for basic Guardian-Loop pipeline
Run training with existing datasets, no adversarial testing
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.prepare_safety_data import SafetyDatasetPreparer
from src.martian.training_integration import MartianTrainingHelper


def test_basic_pipeline():
    """Test the basic pipeline without adversarial components"""
    
    print("🧪 Guardian-Loop Basic Pipeline Test")
    print("=" * 60)
    
    # Step 1: Prepare dataset with real data
    print("\n1️⃣ Preparing dataset with state-of-the-art safety datasets...")
    print("   Using WildGuard, BeaverTails, ToxicChat, WildChat, and more")
    print("   Default balance: 50/50 safe/unsafe for all splits")
    print("   Including safe mutations of unsafe prompts for better boundary learning")
    
    preparer = SafetyDatasetPreparer()
    dataset = preparer.prepare_dataset(
        output_dir="./data/prepared",
        test_split=0.2,
        val_split=0.1,
        balance_ratio=0.5,        # 50% safe prompts
        use_mutations=True        # Create safe variations of unsafe prompts
    )
    print("✅ Dataset prepared with balanced splits!")
    
    # Step 2: Martian integration for validation (Optional)
    print("\n2️⃣ Testing Martian integration capabilities...")
    print("   Current Martian usage:")
    print("   - Training: Can get ground truth labels for unlabeled data")
    print("   - Validation: Compare our predictions with Martian's API")
    print("   - Routing: Pre-filter requests to save API costs")
    print("   - Note: Running in mock mode without API key")
    
    martian_helper = MartianTrainingHelper()
    
    # Example: Get labels for a few prompts
    test_prompts = [
        dataset['test'][i]['prompt'] for i in range(min(5, len(dataset['test'])))
    ]
    
    if martian_helper.client:
        print("\n   Getting Martian labels for test prompts...")
        martian_labels = martian_helper.get_labels_for_prompts(test_prompts)
        print(f"   Received {len(martian_labels)} labels from Martian")
    else:
        print("\n   ⚠️  No Martian API key found - using mock mode")
        print("   In production, Martian would provide:")
        print("   - High-quality ground truth labels")
        print("   - Validation of our safety predictions")
        print("   - Cost-effective routing decisions")
    
    # Step 3: Train safety judge (would happen here)
    print("\n3️⃣ Ready to train safety judge...")
    print("   Would train on:")
    print(f"   - {len(dataset['train'])} training samples")
    print(f"   - {len(dataset['val'])} validation samples")
    print(f"   - {len(dataset['test'])} test samples")
    print("   Training code: src/train_safety_judge.py")
    
    # Step 4: Evaluate without adversarial testing
    print("\n4️⃣ Evaluation (without Rainbow adversarial)...")
    print("   After training, you can evaluate with:")
    print("   - Basic accuracy metrics")
    print("   - AUC-ROC scores")
    print("   - Per-category performance")
    print("   - Martian API validation (if available)")
    
    print("\n✅ Basic pipeline test complete!")
    print("\n📝 Next steps:")
    print("1. Set MARTIAN_API_KEY in .env for real API integration")
    print("2. Run training: python src/train_safety_judge.py")
    print("3. Evaluate: python src/evaluate_safety.py")
    print("4. When ready, add Rainbow adversarial: python src/adversarial/rainbow_loop.py")


if __name__ == "__main__":
    test_basic_pipeline() 