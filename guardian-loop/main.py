"""
Guardian-Loop Main Pipeline
Entry point for the complete implementation
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.prepare_safety_data import SafetyDatasetPreparer
from src.models.safety_judge import SafetyJudgeConfig
from src.train_safety_judge import main as train_main
from src.adversarial.rainbow_loop import RainbowAdversarialLoop
from src.martian.integration import MartianIntegration
from transformers import AutoTokenizer


def setup_environment():
    """Setup the environment and directories"""
    dirs = [
        "data/cache",
        "data/prepared",
        "outputs/checkpoints",
        "outputs/visualizations",
        "outputs/rainbow",
        "cache"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")


def prepare_data():
    """Prepare the safety dataset"""
    print("\n📊 Preparing safety dataset...")
    
    preparer = SafetyDatasetPreparer()
    dataset = preparer.prepare_dataset(
        output_dir="./data/prepared",
        total_dataset_size=10000,  # Total samples
        train_split=0.8,  # 80% train (8000), 10% val (1000), 10% test (1000)
        balance_ratio=0.5,  # 50/50 safe/unsafe
        use_mutations=False  # Disabled for now
    )
    
    print("✅ Dataset preparation complete")
    return dataset


def train_judge(use_existing=False):
    """Train the safety judge model"""
    
    checkpoint_path = Path("./outputs/checkpoints/best_model.pt")
    
    if use_existing and checkpoint_path.exists():
        print(f"\n✅ Using existing checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    print("\n🧠 Training safety judge...")
    
    # Prepare training arguments - OPTIMIZED FOR 8K DATASET
    train_args = [
        '--data_dir', './data/prepared/safety_dataset',
        '--output_dir', './outputs/checkpoints',
        '--batch_size', '8',  # Reduced for Llama 3.1
        '--gradient_accumulation_steps', '4',  # Effective batch = 32
        '--learning_rate', '2e-4',  # Optimized for stability
        '--num_epochs', '15',  # More epochs for 8K samples
        '--max_length', '512',
        '--freeze_layers', '20',  # Unfreeze last 12 layers
        '--pooling', 'mean',  # Better pooling strategy
    ]
    
    # Save original argv and replace
    original_argv = sys.argv
    sys.argv = ['train_safety_judge.py'] + train_args
    
    try:
        # Run training
        train_main()
    finally:
        # Restore argv
        sys.argv = original_argv
    
    print("✅ Training complete")
    return str(checkpoint_path)


def run_adversarial_testing(model_path):
    """Run adversarial testing with Rainbow loop"""
    print("\n🌈 Running adversarial testing...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    from src.models.safety_judge import SafetyJudge
    model = SafetyJudge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Martian (mock for now)
    martian = MartianIntegration()
    
    # Run Rainbow loop
    rainbow_loop = RainbowAdversarialLoop(
        safety_judge=model,
        tokenizer=tokenizer,
        martian_client=martian.client,
        output_dir="./outputs/rainbow"
    )
    
    results = rainbow_loop.run(
        n_iterations=100,  # Reduced for hackathon
        retrain_interval=50,
        visualize_interval=25
    )
    
    print("\n📊 Adversarial Testing Results:")
    print(json.dumps(results['summary'], indent=2))
    
    # Save results
    results_path = Path("./outputs/rainbow/results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")
    return results


def create_demo_package(model_path):
    """Create a demo package with all necessary files"""
    print("\n📦 Creating demo package...")
    
    demo_dir = Path("./guardian_loop_demo")
    demo_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        ("demo.py", "demo.py"),
        ("requirements.txt", "requirements.txt"),
        (model_path, "model/best_model.pt"),
        ("outputs/rainbow/final_archive.json", "data/rainbow_archive.json"),
    ]
    
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = demo_dir / dst
        
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied {src} → {dst}")
    
    # Create README
    readme_content = """# Guardian-Loop Demo

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit demo:
```bash
streamlit run demo.py
```

3. Or run CLI demo:
```bash
python demo.py --mode all
```

## Features

- 🔍 **Safety Analysis**: Test prompts for safety violations
- 🧠 **Mechanistic Interpretability**: Visualize decision-making process
- 🌈 **Adversarial Testing**: Discover model vulnerabilities
- 🚀 **Martian Integration**: Efficient router pre-filtering

Built for the Apart x Martian Hackathon 🏆
"""
    
    with open(demo_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\n✅ Demo package created in {demo_dir}")


def main():
    """Main pipeline entry point"""
    parser = argparse.ArgumentParser(description="Guardian-Loop Pipeline")
    parser.add_argument('--stage', choices=['all', 'setup', 'data', 'train', 'test', 'demo'],
                       default='all', help='Pipeline stage to run')
    parser.add_argument('--use-existing-model', action='store_true',
                       help='Use existing model checkpoint if available')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version for testing')
    
    args = parser.parse_args()
    
    print("🛡️  Guardian-Loop Pipeline")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run stages
    if args.stage in ['all', 'setup']:
        setup_environment()
    
    if args.stage in ['all', 'data']:
        prepare_data()
    
    model_path = None
    if args.stage in ['all', 'train']:
        model_path = train_judge(use_existing=args.use_existing_model)
    elif args.stage in ['test', 'demo']:
        # Find existing model
        model_path = "./outputs/checkpoints/best_model.pt"
        if not Path(model_path).exists():
            print("❌ No trained model found. Please run training first.")
            return
    
    if args.stage in ['all', 'test'] and model_path:
        run_adversarial_testing(model_path)
    
    if args.stage in ['all', 'demo'] and model_path:
        create_demo_package(model_path)
        print("\n🚀 To run the demo:")
        print("   cd guardian_loop_demo")
        print("   streamlit run demo.py")
    
    print("\n✨ Pipeline complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 