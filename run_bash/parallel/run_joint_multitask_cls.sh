#!/bin/bash
################################################################################
# Du-IN Joint Multi-Task + E2E Classification Training Script
#
# This script trains the joint multi-task + end-to-end classification model
# combining semantic, visual, acoustic alignment tasks with direct 61-word
# classification. All four losses are trained simultaneously from MAE checkpoint.
# Adjust parameters below for hyperparameter tuning experiments.
################################################################################

# Experiment Target: run in all subjects

# Get the absolute path of this script (MUST be before changing directory)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Basic configuration
SEEDS="42"                    # Random seeds (space-separated for multiple runs)
# Subject list - will loop through all subjects sequentially
# ALL_SUBJS=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012")
ALL_SUBJS=("001")

# Learning rate schedule
LR_MIN=1e-5                   # Minimum learning rate (cosine annealing end)
LR_MAX=2e-4                   # Maximum learning rate (after warmup)

# Training schedule
N_EPOCHS=500                  # Total number of training epochs
WARMUP_EPOCHS=20              # Number of warmup epochs (linear warmup)
BATCH_SIZE=64                 # Batch size for training

################################################################################
# Joint Multi-Task + E2E Classification Parameters
################################################################################

# Task weights for loss combination (only used when use_uncertainty_weighting=false)
TASK_WEIGHT_SEMANTIC=1.0      # Weight for semantic alignment task
TASK_WEIGHT_VISUAL=1.0        # Weight for visual alignment task
TASK_WEIGHT_ACOUSTIC=1.0      # Weight for acoustic classification task
TASK_WEIGHT_E2E=2.0           # Weight for end-to-end word classification task (NEW)

# Multi-task learning strategy flags
USE_UNCERTAINTY_WEIGHTING=""  # Set to "--use_uncertainty_weighting" to enable auto-balancing, default as ""
# USE_UNCERTAINTY_WEIGHTING="--use_uncertainty_weighting"  # Set to "--use_uncertainty_weighting" to enable auto-balancing, default as ""
ACOUSTIC_USE_CONTRA=""        # Set to "--acoustic_use_contra" to add contrastive loss to acoustic task, default as ""

# Task-specific mapping matrices flag (NEW)
USE_TASK_MAPPING="--use_task_mapping"            # Set to "--use_task_mapping" to enable task-specific mapping with residual, default as ""

################################################################################
# Loss Scales for Each Task
################################################################################

# Semantic alignment loss scales
SEMANTIC_ALIGN_LOSS_SCALE=5.0      # Semantic alignment (MSE) loss scale
SEMANTIC_CONTRA_LOSS_SCALE=0.5     # Semantic contrastive loss scale

# Visual alignment loss scales
VISUAL_ALIGN_LOSS_SCALE=5.0        # Visual alignment (MSE) loss scale
VISUAL_CONTRA_LOSS_SCALE=0.5       # Visual contrastive loss scale

# Acoustic classification loss scales
ACOUSTIC_CLS_LOSS_SCALE=1.0        # Acoustic classification (cross-entropy) loss scale
ACOUSTIC_CONTRA_LOSS_SCALE=0.5     # Acoustic contrastive loss scale (only if acoustic_use_contra is set)

# E2E classification loss scale (NEW)
CLS_LOSS_SCALE=1.0                 # E2E word classification loss scale

################################################################################
# Task-Specific Head Architectures
################################################################################

# Semantic alignment head architecture
SEMANTIC_D_HIDDEN="1024,768"  # Hidden layer dimensions (comma-separated)
SEMANTIC_DROPOUT=0.1               # Dropout rate for semantic head

# Visual alignment head architecture
VISUAL_D_HIDDEN="1024,768"    # Hidden layer dimensions (comma-separated)
VISUAL_DROPOUT=0.1                 # Dropout rate for visual head

# Acoustic classification head architecture
ACOUSTIC_D_HIDDEN="128"            # Hidden layer dimensions (comma-separated)
ACOUSTIC_DROPOUT=0.5               # Dropout rate for acoustic heads

# Fusion head architecture (NEW)
# FUSION_D_HIDDEN="512,256"          # Fusion MLP hidden dimensions (comma-separated)
FUSION_D_HIDDEN=""          # Fusion MLP hidden dimensions (comma-separated)
FUSION_DROPOUT=0.3                 # Dropout rate for fusion head

################################################################################
# Encoder Configuration (Shared Across All Tasks)
################################################################################

# Encoder dropout rates (middle ground for multi-task learning)
ATTN_DROPOUT=0.25              # Attention dropout rate
FF_DROPOUT="0.25,0.1"          # Feedforward dropout rates (comma-separated: [layer1, layer2])

# Contrastive learning parameters (shared by alignment tasks)
CONTRA_D_HIDDEN=32             # Contrastive projection dimension
CONTRA_LOSS_MODE="clip_orig"   # Contrastive loss mode: [clip, clip_orig, unicl]

# Encoder architecture
N_BLOCKS=8                     # Number of transformer blocks
N_HEADS=8                      # Number of attention heads

################################################################################
# Run training
################################################################################

# Change to training directory
cd train/duin

# Loop through all subjects sequentially
for SUBJ in "${ALL_SUBJS[@]}"; do
    echo "========================================="
    echo "Training joint multi-task + e2e classification model for subject ${SUBJ}..."
    echo "========================================="

    # Set subject-specific pretrained checkpoint (from MAE, NOT multitask)
    PT_CKPT="./pretrains/duin/${SUBJ}/mae/model/checkpoint-399.pth"

    python run_joint_multitask_cls.py \
        --seeds ${SEEDS} \
        --subjs ${SUBJ} \
        --subj_idxs 0 \
        --pt_ckpt ${PT_CKPT} \
        --lr_min ${LR_MIN} \
        --lr_max ${LR_MAX} \
        --n_epochs ${N_EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --task_weight_semantic ${TASK_WEIGHT_SEMANTIC} \
        --task_weight_visual ${TASK_WEIGHT_VISUAL} \
        --task_weight_acoustic ${TASK_WEIGHT_ACOUSTIC} \
        --task_weight_e2e ${TASK_WEIGHT_E2E} \
        ${USE_UNCERTAINTY_WEIGHTING} \
        ${ACOUSTIC_USE_CONTRA} \
        --semantic_align_loss_scale ${SEMANTIC_ALIGN_LOSS_SCALE} \
        --semantic_contra_loss_scale ${SEMANTIC_CONTRA_LOSS_SCALE} \
        --visual_align_loss_scale ${VISUAL_ALIGN_LOSS_SCALE} \
        --visual_contra_loss_scale ${VISUAL_CONTRA_LOSS_SCALE} \
        --acoustic_cls_loss_scale ${ACOUSTIC_CLS_LOSS_SCALE} \
        --acoustic_contra_loss_scale ${ACOUSTIC_CONTRA_LOSS_SCALE} \
        --cls_loss_scale ${CLS_LOSS_SCALE} \
        --semantic_d_hidden ${SEMANTIC_D_HIDDEN} \
        --semantic_dropout ${SEMANTIC_DROPOUT} \
        --visual_d_hidden ${VISUAL_D_HIDDEN} \
        --visual_dropout ${VISUAL_DROPOUT} \
        --acoustic_d_hidden ${ACOUSTIC_D_HIDDEN} \
        --acoustic_dropout ${ACOUSTIC_DROPOUT} \
        --fusion_d_hidden "${FUSION_D_HIDDEN}" \
        --fusion_dropout ${FUSION_DROPOUT} \
        --attn_dropout ${ATTN_DROPOUT} \
        --ff_dropout ${FF_DROPOUT} \
        --contra_d_hidden ${CONTRA_D_HIDDEN} \
        --contra_loss_mode ${CONTRA_LOSS_MODE} \
        --n_blocks ${N_BLOCKS} \
        --n_heads ${N_HEADS} \
        ${USE_TASK_MAPPING} \
        --run_script "${SCRIPT_PATH}"

    echo "Finished training subject ${SUBJ}"
    echo ""
done

echo "========================================="
echo "All subjects joint multi-task + e2e classification training completed!"
echo "========================================="

################################################################################
# Example: Multi-seed training for statistical evaluation
################################################################################
# Uncomment below to run with multiple seeds:
# SEEDS="42 123 456 789 1024"

################################################################################
# Example: Enable uncertainty-based automatic task weighting
################################################################################
# Uncomment below to enable uncertainty weighting (auto-balances all 4 tasks):
# USE_UNCERTAINTY_WEIGHTING="--use_uncertainty_weighting"

################################################################################
# Example: Add contrastive loss to acoustic task
################################################################################
# Uncomment below to add contrastive loss to acoustic task:
# ACOUSTIC_USE_CONTRA="--acoustic_use_contra"

################################################################################
# Example: Enable task-specific mapping matrices
################################################################################
# Uncomment below to enable trainable mapping matrices with residual connections:
# USE_TASK_MAPPING="--use_task_mapping"

################################################################################
# Example: Experiment with different task weights
################################################################################
# Experiment 1: Emphasize e2e classification task
# TASK_WEIGHT_SEMANTIC=1.0 TASK_WEIGHT_VISUAL=1.0 TASK_WEIGHT_ACOUSTIC=1.0 TASK_WEIGHT_E2E=3.0

# Experiment 2: Balance all four tasks equally
# TASK_WEIGHT_SEMANTIC=1.0 TASK_WEIGHT_VISUAL=1.0 TASK_WEIGHT_ACOUSTIC=1.0 TASK_WEIGHT_E2E=1.0

# Experiment 3: Lower weight on alignment tasks, higher on classification
# TASK_WEIGHT_SEMANTIC=0.5 TASK_WEIGHT_VISUAL=0.5 TASK_WEIGHT_ACOUSTIC=1.0 TASK_WEIGHT_E2E=2.0

################################################################################
# Example: Experiment with deeper fusion head
################################################################################
# Experiment: Deeper fusion MLP for better classification
# FUSION_D_HIDDEN="1024,512,256"

# Experiment: Simpler fusion head (direct classification)
# FUSION_D_HIDDEN="256"

################################################################################
# Example: Adjust loss scales
################################################################################
# Experiment: Increase e2e classification loss scale
# CLS_LOSS_SCALE=2.0

# Experiment: Decrease alignment loss scales to focus on classification
# SEMANTIC_ALIGN_LOSS_SCALE=2.0 VISUAL_ALIGN_LOSS_SCALE=2.0 CLS_LOSS_SCALE=1.5
