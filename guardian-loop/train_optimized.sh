#!/bin/bash

# Optimized training script for 8K safety dataset with Llama 3.1

echo "🚀 Starting optimized training for Guardian-Loop Safety Judge"
echo "Dataset: 8K samples (4K safe, 4K unsafe)"
echo "Model: Llama 3.1 8B with last 12 layers unfrozen"

# Check if visualization flag is passed
VISUALIZE_FLAG=""
if [[ "$1" == "--visualize" ]]; then
    VISUALIZE_FLAG="--visualize_during_training --visualization_interval 3"
    echo "📊 MI visualizations will be created during training"
fi

# Training with optimized hyperparameters
python -m src.train_safety_judge \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_epochs 15 \
    --freeze_layers 20 \
    --pooling mean \
    --max_length 512 \
    --output_dir ./outputs/checkpoints_optimized \
    $VISUALIZE_FLAG

echo "✅ Training complete!"

if [[ "$1" == "--visualize" ]]; then
    echo "📊 Training visualizations saved to: ./outputs/checkpoints_optimized/training_visualizations/"
    echo "   Open the HTML files to see how the model's behavior evolved!"
fi 