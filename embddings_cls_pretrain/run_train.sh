#!/bin/bash

# ============================================================================
# Embeddings Classification Training Script
# ============================================================================
# This script trains a classifier on concatenated visual + semantic embeddings
# for Chinese word reading classification (61 classes).
#
# Supported Models:
#   - MLP: Multi-layer perceptron with fully connected layers
#   - CNN1D: 1D convolutional network treating embeddings as sequences
#   - CNN2D: 2D convolutional network treating embeddings as 32x48 images
#
# Quick Start:
#   bash run_train.sh                    # Train with MLP (default)
#   bash run_train.sh --model_type cnn1d # Train with 1D CNN
#   bash run_train.sh --model_type cnn2d # Train with 2D CNN
#   bash run_train.sh --help             # Show full help
# ============================================================================

# ============================================================================
# Default Configuration
# ============================================================================

# Data paths
VISUAL_PATH="/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Visual_GT_VitPerchar.npz"
SEMANTIC_PATH="/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Semantic_GT_bert.npz"
OUTPUT_DIR="embddings_cls_pretrain/output"

# Data augmentation
NOISE_TYPE="gaussian"  # Options: gaussian, uniform, salt_pepper
NOISE_LEVEL=0.2        # Noise intensity (std for gaussian, epsilon for uniform, prob for salt_pepper)
N_AUGMENT=500           # Number of augmented samples per original embedding

# Model architecture
MODEL_TYPE="cnn1d"     # Options: mlp, cnn1d, cnn2d

# MLP-specific parameters (only used when MODEL_TYPE=mlp)
MLP_HIDDEN="1024 512 256"  # Hidden layer sizes for MLP

# CNN-specific parameters (only used when MODEL_TYPE=cnn1d or cnn2d)
CNN_CHANNELS=""        # Output channels per layer. Auto-set if empty:
                       #   CNN1D default: [64, 128, 256, 512]
                       #   CNN2D default: [32, 64, 128, 256]
CNN_KERNELS=""         # Kernel sizes per layer. Auto-set if empty:
                       #   CNN1D default: [7, 5, 5, 3]
                       #   CNN2D default: [3, 3, 3, 3]

# Regularization
DROPOUT=0.1            # Dropout rate for MLP
CNN_DROPOUT=0.1        # Dropout rate for CNN models

# Training hyperparameters
BATCH_SIZE=64
EPOCHS=150
LR=5e-4                # Learning rate
WEIGHT_DECAY=1e-4      # Weight decay for AdamW optimizer
SEED=42                # Random seed for reproducibility

# Visualization
VISUALIZE="--visualize"  # Generate t-SNE visualization (comment out to disable)

# ============================================================================
# Command-line Argument Parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --visual_path)
            VISUAL_PATH="$2"
            shift 2
            ;;
        --semantic_path)
            SEMANTIC_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --noise_type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --noise_level)
            NOISE_LEVEL="$2"
            shift 2
            ;;
        --n_augment)
            N_AUGMENT="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --mlp_hidden)
            MLP_HIDDEN="$2"
            shift 2
            ;;
        --cnn_channels)
            CNN_CHANNELS="$2"
            shift 2
            ;;
        --cnn_kernels)
            CNN_KERNELS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --cnn_dropout)
            CNN_DROPOUT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        --help)
            echo "============================================================================"
            echo "Embeddings Classification Training - Help"
            echo "============================================================================"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --visual_path PATH        Path to visual embeddings .npz file"
            echo "  --semantic_path PATH      Path to semantic embeddings .npz file"
            echo "  --output_dir PATH         Output directory for checkpoints and logs"
            echo "  --noise_type TYPE         Type of noise: gaussian, uniform, salt_pepper"
            echo "  --noise_level FLOAT       Noise intensity (default: 0.3)"
            echo "  --n_augment INT           Number of augmented samples per original (default: 50)"
            echo "  --model_type TYPE         Classifier type: mlp, cnn1d, cnn2d (default: mlp)"
            echo "  --mlp_hidden \"SIZES\"      MLP hidden layer sizes (default: \"1024 256\")"
            echo "  --cnn_channels \"SIZES\"   CNN output channels (default: auto-set per model_type)"
            echo "  --cnn_kernels \"SIZES\"    CNN kernel sizes (default: auto-set per model_type)"
            echo "  --dropout FLOAT           Dropout rate for MLP (default: 0.1)"
            echo "  --cnn_dropout FLOAT       Dropout rate for CNN (default: 0.1)"
            echo "  --batch_size INT          Batch size (default: 64)"
            echo "  --epochs INT              Number of epochs (default: 150)"
            echo "  --lr FLOAT                Learning rate (default: 5e-4)"
            echo "  --weight_decay FLOAT      Weight decay (default: 1e-4)"
            echo "  --seed INT                Random seed (default: 42)"
            echo "  --visualize               Generate t-SNE visualization"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  # MLP training (default)"
            echo "  bash $0"
            echo ""
            echo "  # 1D CNN with default architecture (4 layers)"
            echo "  bash $0 --model_type cnn1d"
            echo ""
            echo "  # 2D CNN with default architecture (4 layers)"
            echo "  bash $0 --model_type cnn2d"
            echo ""
            echo "  # 1D CNN with custom architecture (5 layers)"
            echo "  bash $0 --model_type cnn1d --cnn_channels \"64 128 256 512 1024\" --cnn_kernels \"7 5 5 3 3\""
            echo ""
            echo "  # 2D CNN with custom architecture and higher dropout"
            echo "  bash $0 --model_type cnn2d --cnn_channels \"32 64 128\" --cnn_kernels \"5 3 3\" --cnn_dropout 0.2"
            echo ""
            echo "  # Training with custom noise settings"
            echo "  bash $0 --noise_type uniform --noise_level 0.05 --n_augment 20"
            echo ""
            echo "Model Type Details:"
            echo "  - MLP: Traditional fully-connected neural network. Fast, simple baseline."
            echo "  - CNN1D: Treats 1536D embedding as a sequence. Good for learning local patterns."
            echo "  - CNN2D: Treats 1536D embedding as 32x48 image. Good for learning spatial structure."
            echo ""
            echo "============================================================================"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Configuration Summary
# ============================================================================
echo "============================================================================"
echo "Embeddings Classification Training"
echo "============================================================================"
echo "Visual path:    $VISUAL_PATH"
echo "Semantic path:  $SEMANTIC_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Noise type:     $NOISE_TYPE"
echo "Noise level:    $NOISE_LEVEL"
echo "N augment:      $N_AUGMENT"
echo "Model type:     $MODEL_TYPE"
echo "MLP hidden:     $MLP_HIDDEN"
echo "CNN channels:   $([ -z "$CNN_CHANNELS" ] && echo "Auto" || echo "$CNN_CHANNELS")"
echo "CNN kernels:    $([ -z "$CNN_KERNELS" ] && echo "Auto" || echo "$CNN_KERNELS")"
echo "Dropout (MLP):  $DROPOUT"
echo "Dropout (CNN):  $CNN_DROPOUT"
echo "Batch size:     $BATCH_SIZE"
echo "Epochs:         $EPOCHS"
echo "Learning rate:  $LR"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Seed:           $SEED"
echo "Visualize:      $([ -z \"$VISUALIZE\" ] && echo \"No\" || echo \"Yes\")"
echo "============================================================================"
echo ""

# ============================================================================
# Run Training
# ============================================================================
python embddings_cls_pretrain/train_emb_cls.py \
    --visual_path "$VISUAL_PATH" \
    --semantic_path "$SEMANTIC_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --noise_type "$NOISE_TYPE" \
    --noise_level "$NOISE_LEVEL" \
    --n_augment "$N_AUGMENT" \
    --model_type "$MODEL_TYPE" \
    $([ -n "$MLP_HIDDEN" ] && echo "--mlp_hidden $MLP_HIDDEN") \
    $([ -n "$CNN_CHANNELS" ] && echo "--cnn_channels $CNN_CHANNELS") \
    $([ -n "$CNN_KERNELS" ] && echo "--cnn_kernels $CNN_KERNELS") \
    --dropout "$DROPOUT" \
    --cnn_dropout "$CNN_DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    $VISUALIZE

# ============================================================================
# Check Training Result
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "Training failed with exit code $?"
    echo "============================================================================"
    exit 1
fi
