#!/bin/bash

# Embeddings Classification Training Script
# This script runs the embeddings classification training with configurable parameters

# Default parameters
VISUAL_PATH="/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Visual_GT_VitPerchar.npz"
SEMANTIC_PATH="/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Semantic_GT_bert.npz"
OUTPUT_DIR="embddings_cls_pretrain/output"
NOISE_TYPE="gaussian"
NOISE_LEVEL=0.15
N_AUGMENT=50
MLP_HIDDEN=""
DROPOUT=0.1
BATCH_SIZE=64
EPOCHS=100
LR=5e-4
WEIGHT_DECAY=1e-4
SEED=42
VISUALIZE="--visualize"

# Parse command-line arguments
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
        --mlp_hidden)
            MLP_HIDDEN="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --visual_path PATH        Path to visual embeddings .npz file"
            echo "  --semantic_path PATH      Path to semantic embeddings .npz file"
            echo "  --output_dir PATH         Output directory for checkpoints and logs"
            echo "  --noise_type TYPE         Type of noise: gaussian, uniform, salt_pepper"
            echo "  --noise_level FLOAT       Noise intensity (default: 0.1)"
            echo "  --n_augment INT           Number of augmented samples per original (default: 10)"
            echo "  --mlp_hidden \"SIZES\"      Hidden layer sizes (default: \"1024 512 256\")"
            echo "  --dropout FLOAT           Dropout rate (default: 0.1)"
            echo "  --batch_size INT          Batch size (default: 64)"
            echo "  --epochs INT              Number of epochs (default: 100)"
            echo "  --lr FLOAT                Learning rate (default: 1e-3)"
            echo "  --weight_decay FLOAT      Weight decay (default: 1e-4)"
            echo "  --seed INT                Random seed (default: 42)"
            echo "  --visualize               Generate t-SNE visualization"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic training with default parameters"
            echo "  bash $0"
            echo ""
            echo "  # Training with custom noise settings"
            echo "  bash $0 --noise_type uniform --noise_level 0.05 --n_augment 20"
            echo ""
            echo "  # Training with visualization"
            echo "  bash $0 --visualize"
            echo ""
            echo "  # Custom MLP architecture"
            echo "  bash $0 --mlp_hidden \"2048 1024 512\" --dropout 0.2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "====================================="
echo "Embeddings Classification Training"
echo "====================================="
echo "Visual path:    $VISUAL_PATH"
echo "Semantic path:  $SEMANTIC_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Noise type:     $NOISE_TYPE"
echo "Noise level:    $NOISE_LEVEL"
echo "N augment:      $N_AUGMENT"
echo "MLP hidden:     $MLP_HIDDEN"
echo "Dropout:        $DROPOUT"
echo "Batch size:     $BATCH_SIZE"
echo "Epochs:         $EPOCHS"
echo "Learning rate:  $LR"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Seed:           $SEED"
echo "Visualize:      $([ -z \"$VISUALIZE\" ] && echo \"No\" || echo \"Yes\")"
echo "====================================="
echo ""

# Run training
python embddings_cls_pretrain/train_emb_cls.py \
    --visual_path "$VISUAL_PATH" \
    --semantic_path "$SEMANTIC_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --noise_type "$NOISE_TYPE" \
    --noise_level "$NOISE_LEVEL" \
    --n_augment "$N_AUGMENT" \
    $([ -n "$MLP_HIDDEN" ] && echo "--mlp_hidden $MLP_HIDDEN") \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    $VISUALIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "====================================="
else
    echo ""
    echo "====================================="
    echo "Training failed with exit code $?"
    echo "====================================="
    exit 1
fi
