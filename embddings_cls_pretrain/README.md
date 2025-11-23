# Embeddings Classification Pretraining

This module provides tools for training a classification model using augmented visual and semantic embeddings. The training pipeline includes data augmentation with controllable noise, MLP-based classification, and t-SNE visualization.

## Overview

The pipeline consists of:
1. **Data Loading**: Load ground truth visual and semantic embeddings from `.npz` files
2. **Data Augmentation**: Generate augmented samples by adding controllable noise
3. **Classification Training**: Train an MLP classifier on concatenated visual + semantic embeddings
4. **Visualization**: Generate t-SNE plots to visualize the augmented embedding space

## Requirements

All dependencies are already included in the `brain2vec-torch` conda environment:
- PyTorch
- NumPy
- scikit-learn
- matplotlib

## Usage

### Basic Training

```bash
cd embddings_cls_pretrain

# Train with default parameters (Gaussian noise, level=0.1, 10 augmentations)
python train_emb_cls.py

# Train with custom noise type and level
python train_emb_cls.py --noise_type uniform --noise_level 0.05 --n_augment 20

# Train with visualization
python train_emb_cls.py --visualize
```

### Advanced Options

```bash
# Custom MLP architecture with 3 hidden layers
python train_emb_cls.py --mlp_hidden 2048 1024 512 --dropout 0.2

# Long training with larger batch size
python train_emb_cls.py --epochs 200 --batch_size 128 --lr 5e-4

# Multiple experiments with different noise types
python train_emb_cls.py --noise_type gaussian --noise_level 0.1 --output_dir ./output/gaussian_0.1
python train_emb_cls.py --noise_type uniform --noise_level 0.05 --output_dir ./output/uniform_0.05
python train_emb_cls.py --noise_type salt_pepper --noise_level 0.1 --output_dir ./output/salt_pepper_0.1
```

### Standalone Visualization

After training with `--visualize`, you can also generate visualizations separately:

```bash
# Basic t-SNE visualization
python visualize_embeddings.py output/exp_20250123_120000/augmented_embeddings.npz

# Generate all visualization types (t-SNE, noise distribution, distance statistics)
python visualize_embeddings.py output/exp_20250123_120000/augmented_embeddings.npz --all

# Custom t-SNE parameters
python visualize_embeddings.py output/exp_20250123_120000/augmented_embeddings.npz \
    --perplexity 50 --n_iter 2000 --output_dir ./visualizations
```

## Command-Line Arguments

### train_emb_cls.py

#### Data Paths
- `--visual_path`: Path to visual embeddings (default: `./GT_embeddings/Duin_Visual_GT_VitPerchar.npz`)
- `--semantic_path`: Path to semantic embeddings (default: `./GT_embeddings/Duin_Semantic_GT_bert.npz`)
- `--output_dir`: Output directory for checkpoints and logs (default: `./output`)

#### Augmentation Parameters
- `--noise_type`: Type of noise for augmentation (choices: `gaussian`, `uniform`, `salt_pepper`, default: `gaussian`)
  - **gaussian**: Add Gaussian noise N(0, σ²) where σ = noise_level
  - **uniform**: Add uniform noise U(-ε, ε) where ε = noise_level
  - **salt_pepper**: Randomly zero out dimensions with probability = noise_level
- `--noise_level`: Noise intensity (default: `0.1`)
- `--n_augment`: Number of augmented samples per original embedding (default: `10`)

#### Model Architecture
- `--mlp_hidden`: Hidden layer sizes for MLP (space-separated, default: `1024 512 256`)
- `--dropout`: Dropout rate (default: `0.1`)

#### Training Parameters
- `--batch_size`: Batch size (default: `64`)
- `--epochs`: Number of training epochs (default: `100`)
- `--lr`: Learning rate (default: `1e-3`)
- `--weight_decay`: Weight decay for AdamW optimizer (default: `1e-4`)
- `--seed`: Random seed for reproducibility (default: `42`)

#### Visualization
- `--visualize`: Generate t-SNE visualization of augmented embeddings (flag)

### visualize_embeddings.py

- `embeddings_path`: Path to `augmented_embeddings.npz` file (positional argument)
- `--output_dir`: Output directory for figures (default: same as input)
- `--perplexity`: t-SNE perplexity parameter (default: `30`)
- `--n_iter`: Number of t-SNE iterations (default: `1000`)
- `--seed`: Random seed for t-SNE (default: `42`)
- `--all`: Generate all visualization types (flag)

## Output Structure

After training, the output directory will contain:

```
output/
└── exp_YYYYMMDD_HHMMSS/
    ├── config.json                    # Training configuration
    ├── best_model.pth                 # Best model checkpoint (highest val accuracy)
    ├── final_model.pth                # Final model checkpoint (last epoch)
    ├── history.npz                    # Training history (loss, accuracy per epoch)
    ├── test_results.txt               # Test evaluation results
    ├── augmented_embeddings.npz       # Augmented embeddings (if --visualize)
    └── tsne_visualization.png         # t-SNE plot (if --visualize)
```

### Output Files Description

- **config.json**: All hyperparameters and settings used for the experiment
- **best_model.pth**: Model checkpoint with highest validation accuracy
  - Keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_acc`, `config`
- **final_model.pth**: Model checkpoint from the last training epoch
- **history.npz**: NumPy archive with training metrics
  - Arrays: `train_loss`, `train_acc`, `val_loss`, `val_acc`
- **test_results.txt**: Test set evaluation including classification report
- **augmented_embeddings.npz**: Saved augmented embeddings for later visualization
  - Arrays: `visual`, `semantic`, `labels`, `n_augment`, `noise_type`, `noise_level`

## Architecture Details

### Input Features
- **Visual embeddings**: 768-dimensional (from ViT per-character encoder)
- **Semantic embeddings**: 768-dimensional (from BERT)
- **Concatenated features**: 1536-dimensional input to MLP

### MLP Classifier Structure

```
Input (1536-dim)
    ↓
Linear(1536 → hidden[0]) → BatchNorm → ReLU → Dropout
    ↓
Linear(hidden[0] → hidden[1]) → BatchNorm → ReLU → Dropout
    ↓
... (repeat for all hidden layers)
    ↓
Linear(hidden[-1] → 61) → Softmax
    ↓
Output (61 classes)
```

### Data Augmentation Process

For each original embedding:
1. Keep the original sample
2. Generate `n_augment` noisy versions
3. Total dataset size: 61 × (1 + n_augment)

**Example with n_augment=10:**
- Original samples: 61
- Augmented samples: 61 × 10 = 610
- Total samples: 671

### Train/Val/Test Split

- Training: 70%
- Validation: 15%
- Test: 15%
- **Stratified split** ensures balanced class distribution across splits

## Visualization Options

The `visualize_embeddings.py` script provides three types of visualizations:

1. **t-SNE Visualization** (default)
   - 2D projection of high-dimensional embeddings
   - Shows clustering and separation of classes
   - Distinguishes original vs augmented samples

2. **Noise Distribution** (`--all` flag)
   - Histogram of noise values added to embeddings
   - Shows statistical properties of augmentation
   - Verifies noise follows expected distribution

3. **Distance Statistics** (`--all` flag)
   - L2 distance between original and augmented embeddings
   - Separate plots for visual and semantic embeddings
   - Helps assess augmentation strength

## Example Workflow

### Experiment 1: Compare Noise Types

```bash
# Train with different noise types
python train_emb_cls.py --noise_type gaussian --noise_level 0.1 --output_dir ./output/gaussian --visualize
python train_emb_cls.py --noise_type uniform --noise_level 0.1 --output_dir ./output/uniform --visualize
python train_emb_cls.py --noise_type salt_pepper --noise_level 0.1 --output_dir ./output/salt_pepper --visualize

# Compare test accuracies
grep "Test Accuracy" output/*/test_results.txt
```

### Experiment 2: Tune Noise Level

```bash
# Train with different noise levels
for level in 0.05 0.1 0.15 0.2; do
    python train_emb_cls.py --noise_type gaussian --noise_level $level \
        --output_dir ./output/noise_${level}
done

# Plot results
grep "Test Accuracy" output/noise_*/test_results.txt
```

### Experiment 3: Optimize MLP Architecture

```bash
# Small MLP
python train_emb_cls.py --mlp_hidden 512 256 --output_dir ./output/mlp_small

# Medium MLP (default)
python train_emb_cls.py --mlp_hidden 1024 512 256 --output_dir ./output/mlp_medium

# Large MLP
python train_emb_cls.py --mlp_hidden 2048 1024 512 256 --output_dir ./output/mlp_large
```

## Tips for Best Results

1. **Start with default parameters** to establish a baseline
2. **Use `--visualize`** to understand the effect of augmentation
3. **Experiment with noise_level** between 0.05 and 0.2
4. **Try different noise types** - each has different characteristics:
   - Gaussian: Smooth, continuous perturbations
   - Uniform: Bounded, uniform perturbations
   - Salt & Pepper: Sparse, discrete perturbations
5. **Adjust n_augment** based on overfitting:
   - Increase if model overfits (large train/val gap)
   - Decrease if training is too slow
6. **Monitor validation accuracy** - early stopping is automatic
7. **Use multiple seeds** for robust evaluation:
   ```bash
   for seed in 42 123 456 789 2024; do
       python train_emb_cls.py --seed $seed --output_dir ./output/seed_${seed}
   done
   ```

## Troubleshooting

### Issue: ImportError for visualize_embeddings

**Solution**: Run from the `embddings_cls_pretrain/` directory:
```bash
cd embddings_cls_pretrain
python train_emb_cls.py --visualize
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or model size:
```bash
python train_emb_cls.py --batch_size 32 --mlp_hidden 512 256
```

### Issue: Slow t-SNE computation

**Solution**: Reduce perplexity or iterations:
```bash
python visualize_embeddings.py data.npz --perplexity 15 --n_iter 500
```

### Issue: Poor test accuracy

**Potential causes and solutions**:
1. Noise level too high → Reduce `--noise_level`
2. Insufficient augmentation → Increase `--n_augment`
3. Model too simple → Add more hidden layers or increase sizes
4. Learning rate too high → Reduce `--lr`
5. Overfitting → Increase `--dropout` or reduce model complexity

## Expected Performance

With default parameters (Gaussian noise, level=0.1, 10 augmentations):
- **Training accuracy**: ~95-98% (epoch 100)
- **Validation accuracy**: ~90-95%
- **Test accuracy**: ~88-93%

*Note: Actual performance may vary based on random seed and augmentation.*

## Citation

If you use this code in your research, please cite the Du-IN paper:

```bibtex
@inproceedings{du2024duin,
  title={Du-IN: Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals},
  author={Du, Xiaochen and others},
  booktitle={NeurIPS},
  year={2024}
}
```
