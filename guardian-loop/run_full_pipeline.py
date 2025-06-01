#!/usr/bin/env python3
"""
Run the complete Guardian-Loop pipeline (without Open-Ended adversarial)
Includes: data prep, training, evaluation, MI visualizations
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse
from dotenv import load_dotenv
from huggingface_hub import login


def run_command(cmd, description):
    """Run a command with nice output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    # Load environment variables and login to HF
    load_dotenv()
    if 'HF_TOKEN' not in os.environ:
        print("⚠️  Error: HF_TOKEN not found in .env file")
        print("Please add your HuggingFace token to .env file:")
        print("HF_TOKEN=your_token_here")
        sys.exit(1)
    
    login(token=os.getenv('HF_TOKEN'))
    print("✅ Successfully logged in to HuggingFace")

    parser = argparse.ArgumentParser(description='Run Guardian-Loop pipeline')
    parser.add_argument('--skip-data-prep', action='store_true', 
                       help='Skip data preparation if already done')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training if model already exists')
    parser.add_argument('--skip-open-ended', action='store_true',
                       help='Skip Open-Ended adversarial testing')
    parser.add_argument('--demo-only', action='store_true',
                       help='Only run the demo')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--visualize-training', action='store_true',
                       help='Create MI visualizations during training')
    parser.add_argument('--run-advanced-mi', action='store_true',
                       help='Run advanced MI analysis during training (slower but more detailed)')
    args = parser.parse_args()
    
    print("""
    🛡️  Guardian-Loop Full Pipeline Runner
    ====================================
    This will run:
    1. Data preparation (with mutations)
    2. Safety judge training (with optional MI visualizations)
    3. Model evaluation
    4. MI visualizations
    5. Interactive demo
    
    Note: Open-Ended adversarial testing is skipped
    """)
    
    if args.skip_open_ended:
        print("    ⚠️  Open-Ended adversarial testing: SKIPPED")
    else:
        print("    6. Open-Ended adversarial testing")
    
    if args.run_advanced_mi:
        print("    🧠 Advanced MI analysis: ENABLED (will be slower)")
    
    print("")
    
    # Check for API key
    if not os.getenv('MARTIAN_API_KEY'):
        print("⚠️  Warning: No MARTIAN_API_KEY found")
        print("   - Mutations will use templates instead of GPT-4o")
        print("   - Martian validation will run in mock mode")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    start_time = time.time()
    
    # Step 1: Data Preparation
    if not args.skip_data_prep and not args.demo_only and not args.eval_only:
        # Prepare safety data
        run_command(
            "python guardian-loop/src/data/prepare_safety_data.py",
            "Preparing safety datasets"
        )
        
        # Prepare feasibility data
        run_command(
            "python guardian-loop/src/data/prepare_feasibility_data.py",
            "Preparing feasibility datasets"
        )
        
        print("\n📊 Dataset Statistics:")
        print("Safety Dataset:")
        print("   - Train: data/safety/train.json")
        print("   - Val: data/safety/val.json")
        print("   - Test: data/safety/test.json")
        print("\nFeasibility Dataset:")
        print("   - Train: data/feasibility/train.json")
        print("   - Val: data/feasibility/val.json")
        print("   - Test: data/feasibility/test.json")
    
    # Step 2: Training
    if not args.skip_training and not args.demo_only and not args.eval_only:
        cmd = ("python guardian-loop/src/train_safety_judge.py --batch_size 8 --gradient_accumulation_steps 4 "
               "--learning_rate 2e-4 --num_epochs 15 --freeze_layers 20")
        
        if args.visualize_training:
            cmd += " --visualize_during_training --visualization_interval 1"
        
        if args.run_advanced_mi:
            cmd += " --run_advanced_mi"
            
        run_command(
            cmd,
            "Training safety judge on prepared data (optimized for 8K samples)"
        )
        
        if args.visualize_training:
            print("\n📊 Training visualizations saved to: outputs/checkpoints/training_visualizations/")
            print("   Visualizations are generated for EVERY epoch when --visualize-training is enabled!")
            print("   Open the HTML files to see how the model evolved during training!")
    
    # Step 3: Evaluation
    if not args.demo_only:
        # Basic evaluation
        run_command(
            "python guardian-loop/src/evaluate_safety.py",
            "Evaluating model performance"
        )
        
        # MI visualizations
        print("\n🔬 Generating Mechanistic Interpretability visualizations...")
        
        # Token attribution heatmaps
        run_command(
            "python guardian-loop/src/mi_tools/visualization.py --mode token_attribution --num_samples 10",
            "Creating token attribution heatmaps"
        )
        
        # Attention patterns
        run_command(
            "python guardian-loop/src/mi_tools/visualization.py --mode attention_patterns --num_samples 10",
            "Analyzing attention patterns"
        )
        
        # Layer activations
        run_command(
            "python guardian-loop/src/mi_tools/visualization.py --mode layer_activations --num_samples 10",
            "Visualizing layer activations"
        )
        
        print("\n📁 MI visualizations saved to: outputs/mi_visualizations/")
    
    # Step 4: Martian Integration Test
    if not args.demo_only and not args.eval_only:
        run_command(
            "python guardian-loop/src/martian/test_integration.py",
            "Testing Martian API integration"
        )
    
    # Step 5: Interactive Demo
    if not args.eval_only:
        print("\n🎯 Launching interactive demo...")
        print("   This includes:")
        print("   - Safety analysis interface")
        print("   - MI visualization viewer")
        print("   - Martian routing demo")
        print("   - Performance metrics")
        
        # Check if streamlit is installed
        try:
            import streamlit
            run_command(
                "streamlit run demo.py",
                "Launching Streamlit demo"
            )
        except ImportError:
            print("⚠️  Streamlit not installed, running CLI demo instead")
            run_command(
                "python demo.py --mode cli",
                "Running CLI demo"
            )
    
    # Step 6: Open-Ended Adversarial Testing (Optional)
    if not args.skip_open_ended and not args.demo_only and not args.eval_only:
        print("\n🌈 Open-Ended Adversarial Testing")
        print("   This will:")
        print("   - Generate adversarial prompts")
        print("   - Test model robustness")
        print("   - Create vulnerability report")
        
        run_command(
            "python guardian-loop/src/adversarial/open_ended_loop.py --iterations 100 --output_dir outputs/open_ended",
            "Running Open-Ended adversarial testing"
        )
        
        print("\n📁 Open-Ended results saved to: outputs/open_ended/")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Pipeline completed in {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    print("\n📊 Key Outputs:")
    print("   - Trained model: outputs/checkpoints/best_model.pt")
    print("   - Evaluation results: outputs/evaluation/")
    print("   - MI visualizations: outputs/mi_visualizations/")
    
    if args.visualize_training:
        print("   - Training MI evolution: outputs/checkpoints/training_visualizations/")
    
    if not args.skip_open_ended:
        print("   - Open-Ended adversarial results: outputs/open_ended/")
    
    print("   - Demo available at: http://localhost:8501")
    
    print("\n🚀 Next Steps:")
    print("   1. Review evaluation metrics")
    print("   2. Explore MI visualizations")
    
    if args.visualize_training:
        print("   3. Check training evolution visualizations")
    
    print("   4. Test with your own prompts in the demo")
    
    if args.skip_open_ended:
        print("   5. When ready: add Open-Ended adversarial with --skip-open-ended removed")


if __name__ == "__main__":
    main() 