#!/bin/bash

# Run Guardian-Loop pipeline without Open-Ended adversarial testing
# Faster iteration for development and testing

echo "🛡️ Guardian-Loop Pipeline (No Open-Ended Testing)"
echo "=============================================="
echo "This will run:"
echo "  1. Data preparation"
echo "  2. Training with MI visualizations"
echo "  3. Evaluation"
echo "  4. Final MI analysis"
echo "  5. Demo"
echo ""
echo "Open-Ended adversarial testing is SKIPPED for faster iteration"
echo ""

# Run the pipeline with visualizations and no Open-Ended
python guardian-loop/run_full_pipeline.py \
    --visualize-training \
    --skip-open-ended \
    --run-advanced-mi

echo ""
echo "✨ Pipeline complete!"
echo ""
echo "📊 Key outputs to check:"
echo "  1. Training progress: outputs/checkpoints/training_visualizations/"
echo "     - See how model behavior evolved during training"
echo "     - Compare safe vs unsafe circuit divergence over epochs"
echo "  2. Final model: outputs/checkpoints/best_model.pt"
echo "  3. Evaluation results: outputs/evaluation/"
echo ""
echo "💡 To run Open-Ended adversarial testing later:"
echo "   python src/adversarial/open_ended_loop.py --iterations 100" 