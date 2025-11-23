#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embeddings Classification Pretraining

Trains a classification model using augmented visual and semantic embeddings.
Supports multiple noise types and configurable MLP architecture.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# Add parent directory to path to import GT_embeddings loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from GT_embeddings.load_GT import load_npz_vis, load_npz_semantic
from datetime import datetime


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(visual_path, semantic_path):
    """Load visual and semantic embeddings from .npz files using GT_embeddings loaders.

    Args:
        visual_path: Path to visual embeddings .npz file
        semantic_path: Path to semantic embeddings .npz file

    Returns:
        visual_emb: (n_words, d_visual) array
        semantic_emb: (n_words, d_semantic) array
        labels: (n_words,) array of integer labels
    """
    # Use the standard loading functions from GT_embeddings/load_GT.py
    chars, visual_emb = load_npz_vis(visual_path)
    words, semantic_emb = load_npz_semantic(semantic_path, type='mean')

    # Generate integer labels (0 to n_words-1)
    labels = np.arange(len(visual_emb))

    print(f"Loaded visual embeddings: {visual_emb.shape}")
    print(f"Loaded semantic embeddings: {semantic_emb.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    return visual_emb, semantic_emb, labels


def add_gaussian_noise(embedding, noise_level):
    """Add Gaussian noise N(0, σ²) to embedding."""
    noise = np.random.normal(0, noise_level, size=embedding.shape)
    return embedding + noise


def add_uniform_noise(embedding, noise_level):
    """Add uniform noise U(-ε, ε) to embedding."""
    noise = np.random.uniform(-noise_level, noise_level, size=embedding.shape)
    return embedding + noise


def add_salt_pepper_noise(embedding, noise_level):
    """Add salt & pepper noise (randomly zero out dimensions)."""
    noisy_embedding = embedding.copy()
    mask = np.random.random(size=embedding.shape) < noise_level
    noisy_embedding[mask] = 0
    return noisy_embedding


def augment_embeddings(visual_emb, semantic_emb, labels, n_augment, noise_type, noise_level):
    """Generate augmented dataset with noise.

    Args:
        visual_emb: (n_words, d_visual) array
        semantic_emb: (n_words, d_semantic) array
        labels: (n_words,) array
        n_augment: Number of augmented samples per original
        noise_type: 'gaussian', 'uniform', or 'salt_pepper'
        noise_level: Noise intensity parameter

    Returns:
        aug_visual: (n_words * (n_augment + 1), d_visual) array
        aug_semantic: (n_words * (n_augment + 1), d_semantic) array
        aug_labels: (n_words * (n_augment + 1),) array
    """
    n_words = len(labels)
    d_visual = visual_emb.shape[1]
    d_semantic = semantic_emb.shape[1]

    # Initialize augmented arrays (original + augmented)
    total_samples = n_words * (n_augment + 1)
    aug_visual = np.zeros((total_samples, d_visual), dtype=np.float32)
    aug_semantic = np.zeros((total_samples, d_semantic), dtype=np.float32)
    aug_labels = np.zeros(total_samples, dtype=np.int64)

    # Select noise function
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'uniform': add_uniform_noise,
        'salt_pepper': add_salt_pepper_noise
    }
    add_noise = noise_functions[noise_type]

    # Generate augmented data
    for i in range(n_words):
        # Original sample
        idx = i * (n_augment + 1)
        aug_visual[idx] = visual_emb[i]
        aug_semantic[idx] = semantic_emb[i]
        aug_labels[idx] = labels[i]

        # Augmented samples
        for j in range(n_augment):
            idx = i * (n_augment + 1) + j + 1
            aug_visual[idx] = add_noise(visual_emb[i], noise_level)
            aug_semantic[idx] = add_noise(semantic_emb[i], noise_level)
            aug_labels[idx] = labels[i]

    print(f"\nAugmented dataset created:")
    print(f"  Original samples: {n_words}")
    print(f"  Augmented samples per original: {n_augment}")
    print(f"  Total samples: {total_samples}")
    print(f"  Noise type: {noise_type}, level: {noise_level}")

    return aug_visual, aug_semantic, aug_labels


class EmbeddingDataset(Dataset):
    """Dataset for concatenated visual + semantic embeddings."""

    def __init__(self, visual_emb, semantic_emb, labels, return_separate=False, normalize=True):
        self.visual_emb = torch.FloatTensor(visual_emb)
        self.semantic_emb = torch.FloatTensor(semantic_emb)
        self.labels = torch.LongTensor(labels)
        self.return_separate = return_separate
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.return_separate:
            # Return separate embeddings for cosine similarity evaluation
            return self.visual_emb[idx], self.semantic_emb[idx], self.labels[idx]
        else:
            # Concatenate visual and semantic embeddings
            features = torch.cat([self.visual_emb[idx], self.semantic_emb[idx]], dim=0)
            # Normalize the concatenated embedding to unit L2 norm (optional)
            if self.normalize:
                features = features / torch.norm(features, p=2)
            return features, self.labels[idx]


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier with configurable architecture."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CNN1DClassifier(nn.Module):
    """1D CNN classifier for processing flattened embeddings as sequences."""

    def __init__(self, input_dim, channels, kernels, output_dim, dropout=0.1):
        super(CNN1DClassifier, self).__init__()
        self.input_dim = input_dim

        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Input reshaped to (batch, 1, input_dim)

        for out_channels, kernel_size in zip(channels, kernels):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global pooling and output layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x):
        # Reshape from (batch, input_dim) to (batch, 1, input_dim)
        x = x.unsqueeze(1)

        # Apply convolutions
        x = self.conv_layers(x)

        # Global pooling: (batch, channels, length) -> (batch, channels, 1)
        x = self.pool(x)

        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)

        # Final classification
        x = self.fc(x)
        return x


class CNN2DClassifier(nn.Module):
    """2D CNN classifier treating embeddings as pseudo-images."""

    def __init__(self, input_dim, channels, kernels, output_dim, dropout=0.1):
        super(CNN2DClassifier, self).__init__()
        self.input_dim = input_dim

        # Compute 2D reshape dimensions (1536 -> 1x32x48)
        # We use height=32, width=48 as 32*48=1536
        self.height = 32
        self.width = 48
        assert self.height * self.width == input_dim, f"Height*Width must equal input_dim: {self.height}*{self.width} != {input_dim}"

        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Input reshaped to (batch, 1, height, width)

        for out_channels, kernel_size in zip(channels, kernels):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global pooling and output layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x):
        # Reshape from (batch, input_dim) to (batch, 1, height, width)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)

        # Apply convolutions
        x = self.conv_layers(x)

        # Global pooling: (batch, channels, H, W) -> (batch, channels, 1, 1)
        x = self.pool(x)

        # Flatten: (batch, channels, 1, 1) -> (batch, channels)
        x = x.view(batch_size, -1)

        # Final classification
        x = self.fc(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def evaluate_cosine_similarity(gt_visual_emb, gt_semantic_emb, dataloader, device):
    """Evaluate using cosine similarity to GT class embeddings.

    Args:
        gt_visual_emb: (n_classes, d_visual) GT visual embeddings for each class
        gt_semantic_emb: (n_classes, d_semantic) GT semantic embeddings for each class
        dataloader: DataLoader with return_separate=True
        device: torch device

    Returns:
        visual_acc: Accuracy using visual embeddings only
        semantic_acc: Accuracy using semantic embeddings only
        concat_acc: Accuracy using concatenated embeddings
    """
    # Convert GT embeddings to torch tensors and move to device
    gt_visual = torch.FloatTensor(gt_visual_emb).to(device)  # (n_classes, d_visual)
    gt_semantic = torch.FloatTensor(gt_semantic_emb).to(device)  # (n_classes, d_semantic)

    # Normalize GT embeddings (they should already be normalized, but ensure it)
    gt_visual = gt_visual / torch.norm(gt_visual, dim=1, keepdim=True)
    gt_semantic = gt_semantic / torch.norm(gt_semantic, dim=1, keepdim=True)

    # Concatenate GT embeddings for concat similarity
    gt_concat = torch.cat([gt_visual, gt_semantic], dim=1)  # (n_classes, d_visual + d_semantic)
    gt_concat = gt_concat / torch.norm(gt_concat, dim=1, keepdim=True)

    visual_preds = []
    semantic_preds = []
    concat_preds = []
    all_labels = []

    with torch.no_grad():
        for visual_emb, semantic_emb, labels in dataloader:
            visual_emb = visual_emb.to(device)  # (batch, d_visual)
            semantic_emb = semantic_emb.to(device)  # (batch, d_semantic)
            labels = labels.to(device)

            # Normalize sample embeddings
            visual_emb = visual_emb / torch.norm(visual_emb, dim=1, keepdim=True)
            semantic_emb = semantic_emb / torch.norm(semantic_emb, dim=1, keepdim=True)

            # Compute visual-only cosine similarity
            # (batch, d_visual) @ (d_visual, n_classes) -> (batch, n_classes)
            visual_sim = torch.matmul(visual_emb, gt_visual.T)
            visual_pred = visual_sim.argmax(dim=1).cpu().numpy()
            visual_preds.extend(visual_pred)

            # Compute semantic-only cosine similarity
            semantic_sim = torch.matmul(semantic_emb, gt_semantic.T)
            semantic_pred = semantic_sim.argmax(dim=1).cpu().numpy()
            semantic_preds.extend(semantic_pred)

            # Compute concatenated cosine similarity
            concat_emb = torch.cat([visual_emb, semantic_emb], dim=1)
            concat_emb = concat_emb / torch.norm(concat_emb, dim=1, keepdim=True)
            concat_sim = torch.matmul(concat_emb, gt_concat.T)
            concat_pred = concat_sim.argmax(dim=1).cpu().numpy()
            concat_preds.extend(concat_pred)

            all_labels.extend(labels.cpu().numpy())

    # Compute accuracies
    visual_acc = accuracy_score(all_labels, visual_preds)
    semantic_acc = accuracy_score(all_labels, semantic_preds)
    concat_acc = accuracy_score(all_labels, concat_preds)

    return visual_acc, semantic_acc, concat_acc


def main(args):
    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {exp_dir}/config.json")

    # Load embeddings
    print("\n=== Loading Embeddings ===")
    visual_emb, semantic_emb, labels = load_embeddings(args.visual_path, args.semantic_path)
    
    #normalized two embeddings
    visual_emb = visual_emb / np.linalg.norm(visual_emb, axis=1, keepdims=True)
    semantic_emb = semantic_emb / np.linalg.norm(semantic_emb, axis=1, keepdims=True)

    # Augment embeddings
    print("\n=== Augmenting Embeddings ===")
    aug_visual, aug_semantic, aug_labels = augment_embeddings(
        visual_emb, semantic_emb, labels,
        n_augment=args.n_augment,
        noise_type=args.noise_type,
        noise_level=args.noise_level
    )

    # Store augmented embeddings for visualization (not saved to disk)
    aug_emb_data = {
        'visual': aug_visual,
        'semantic': aug_semantic,
        'labels': aug_labels,
        'n_augment': args.n_augment,
        'noise_type': args.noise_type,
        'noise_level': args.noise_level
    }

    # Split dataset (stratified)
    print("\n=== Splitting Dataset ===")
    indices = np.arange(len(aug_labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=args.seed, stratify=aug_labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15/0.7, random_state=args.seed, stratify=aug_labels[train_idx]
    )

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # Create datasets and dataloaders
    # For CNN models, don't normalize the concatenated embeddings (they're already normalized individually)
    # For MLP, keep the normalization
    normalize_concat = (args.model_type == 'mlp')
    print(f"Concatenated embedding normalization: {normalize_concat} (model_type={args.model_type})")

    train_dataset = EmbeddingDataset(aug_visual[train_idx], aug_semantic[train_idx], aug_labels[train_idx], normalize=normalize_concat)
    val_dataset = EmbeddingDataset(aug_visual[val_idx], aug_semantic[val_idx], aug_labels[val_idx], normalize=normalize_concat)
    test_dataset = EmbeddingDataset(aug_visual[test_idx], aug_semantic[test_idx], aug_labels[test_idx], normalize=normalize_concat)

    # Create separate datasets for cosine similarity evaluation
    val_dataset_cossim = EmbeddingDataset(aug_visual[val_idx], aug_semantic[val_idx], aug_labels[val_idx], return_separate=True)
    test_dataset_cossim = EmbeddingDataset(aug_visual[test_idx], aug_semantic[test_idx], aug_labels[test_idx], return_separate=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Dataloaders for cosine similarity evaluation
    val_loader_cossim = DataLoader(val_dataset_cossim, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader_cossim = DataLoader(test_dataset_cossim, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    print("\n=== Initializing Model ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = visual_emb.shape[1] + semantic_emb.shape[1]  # 768 + 768 = 1536
    output_dim = len(np.unique(labels))  # 61 classes

    # Initialize model based on model_type
    if args.model_type == 'mlp':
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=args.mlp_hidden,
            output_dim=output_dim,
            dropout=args.dropout
        ).to(device)
        print(f"Model architecture: MLP")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dims: {args.mlp_hidden}")
        print(f"  Output dim: {output_dim}")
        print(f"  Dropout: {args.dropout}")

    elif args.model_type == 'cnn1d':
        # Set default CNN1D architecture if not specified
        cnn_channels = args.cnn_channels if args.cnn_channels else [64, 128, 256, 512]
        cnn_kernels = args.cnn_kernels if args.cnn_kernels else [7, 5, 5, 3]

        model = CNN1DClassifier(
            input_dim=input_dim,
            channels=cnn_channels,
            kernels=cnn_kernels,
            output_dim=output_dim,
            dropout=args.cnn_dropout
        ).to(device)
        print(f"Model architecture: CNN1D")
        print(f"  Input dim: {input_dim}")
        print(f"  Conv channels: {cnn_channels}")
        print(f"  Conv kernels: {cnn_kernels}")
        print(f"  Output dim: {output_dim}")
        print(f"  Dropout: {args.cnn_dropout}")

    elif args.model_type == 'cnn2d':
        # Set default CNN2D architecture if not specified
        cnn_channels = args.cnn_channels if args.cnn_channels else [32, 64, 128, 256]
        cnn_kernels = args.cnn_kernels if args.cnn_kernels else [3, 3, 3, 3]

        model = CNN2DClassifier(
            input_dim=input_dim,
            channels=cnn_channels,
            kernels=cnn_kernels,
            output_dim=output_dim,
            dropout=args.cnn_dropout
        ).to(device)
        print(f"Model architecture: CNN2D")
        print(f"  Input dim: {input_dim} (reshaped to 1x32x48)")
        print(f"  Conv channels: {cnn_channels}")
        print(f"  Conv kernels: {cnn_kernels}")
        print(f"  Output dim: {output_dim}")
        print(f"  Dropout: {args.cnn_dropout}")

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_cossim_visual_acc': [], 'val_cossim_semantic_acc': [], 'val_cossim_concat_acc': []
    }

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Evaluate using cosine similarity
        val_visual_acc, val_semantic_acc, val_concat_acc = evaluate_cosine_similarity(
            visual_emb, semantic_emb, val_loader_cossim, device
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_cossim_visual_acc'].append(val_visual_acc)
        history['val_cossim_semantic_acc'].append(val_semantic_acc)
        history['val_cossim_concat_acc'].append(val_concat_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  CosSim Acc - Visual: {val_visual_acc:.4f}, Semantic: {val_semantic_acc:.4f}, Concat: {val_concat_acc:.4f}")

        # Track best model (not saved to disk)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Training history not saved to disk

    # Test evaluation
    print("\n=== Test Evaluation ===")
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    # Evaluate test set using cosine similarity
    test_visual_acc, test_semantic_acc, test_concat_acc = evaluate_cosine_similarity(
        visual_emb, semantic_emb, test_loader_cossim, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"\nCosine Similarity Accuracies (Test Set):")
    print(f"  Visual-only: {test_visual_acc:.4f}")
    print(f"  Semantic-only: {test_semantic_acc:.4f}")
    print(f"  Concatenated: {test_concat_acc:.4f}")

    # Classification report
    print("\n=== Classification Report ===")
    report = classification_report(test_labels, test_preds, digits=4)
    print(report)

    # Save test results
    with open(os.path.join(exp_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n\n")
        f.write("Cosine Similarity Accuracies (Test Set):\n")
        f.write(f"  Visual-only: {test_visual_acc:.4f}\n")
        f.write(f"  Semantic-only: {test_semantic_acc:.4f}\n")
        f.write(f"  Concatenated: {test_concat_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nResults saved to {exp_dir}")

    # Visualization
    if args.visualize:
        print("\n=== Generating Visualization ===")
        try:
            from visualize_embeddings import visualize_tsne
            # Pass in-memory data instead of file path
            visualize_tsne(
                aug_emb_data,
                os.path.join(exp_dir, 'tsne_visualization.png')
            )
            print(f"t-SNE visualization saved to {exp_dir}/tsne_visualization.png")
        except Exception as e:
            print(f"Failed to generate visualization: {e}")
            print("You can run visualize_embeddings.py manually later.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP classifier on augmented embeddings')

    # Data paths
    parser.add_argument('--visual_path', type=str,
                        default='./GT_embeddings/Duin_Visual_GT_VitPerchar.npz',
                        help='Path to visual embeddings .npz file')
    parser.add_argument('--semantic_path', type=str,
                        default='./GT_embeddings/Duin_Semantic_GT_bert.npz',
                        help='Path to semantic embeddings .npz file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints and logs')

    # Augmentation parameters
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'salt_pepper'],
                        help='Type of noise for augmentation')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise intensity (std for gaussian, epsilon for uniform, prob for salt_pepper)')
    parser.add_argument('--n_augment', type=int, default=10,
                        help='Number of augmented samples per original embedding')

    # Model architecture
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'cnn1d', 'cnn2d'],
                        help='Type of classifier model')
    parser.add_argument('--mlp_hidden', type=int, nargs='*', default=[1024, 512, 256],
                        help='Hidden layer sizes for MLP (space-separated). Empty for linear classifier.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for MLP')
    parser.add_argument('--cnn_channels', type=int, nargs='*', default=None,
                        help='Output channels for each CNN layer. Default: [64,128,256,512] for CNN1D, [32,64,128,256] for CNN2D')
    parser.add_argument('--cnn_kernels', type=int, nargs='*', default=None,
                        help='Kernel sizes for each CNN layer. Default: [7,5,5,3] for CNN1D, [3,3,3,3] for CNN2D')
    parser.add_argument('--cnn_dropout', type=float, default=0.1,
                        help='Dropout rate for CNN models')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate t-SNE visualization of augmented embeddings')

    args = parser.parse_args()
    main(args)
