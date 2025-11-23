#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for augmented embeddings using t-SNE.

Generates 2D t-SNE plots showing original and augmented embeddings.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def visualize_tsne(embeddings_path_or_data, output_path, perplexity=30, n_iter=1000, random_state=42):
    """
    Generate t-SNE visualization of augmented embeddings.

    Args:
        embeddings_path_or_data: Path to augmented_embeddings.npz file OR dict with keys:
                                 'visual', 'semantic', 'labels', 'n_augment', 'noise_type', 'noise_level'
        output_path: Path to save the output figure
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        random_state: Random seed for t-SNE
    """
    # Load augmented embeddings (from file or dict)
    if isinstance(embeddings_path_or_data, dict):
        data = embeddings_path_or_data
        visual = data['visual']
        semantic = data['semantic']
        labels = data['labels']
        n_augment = int(data['n_augment'])
        noise_type = str(data['noise_type'])
        noise_level = float(data['noise_level'])
    else:
        data = np.load(embeddings_path_or_data)
        visual = data['visual']
        semantic = data['semantic']
        labels = data['labels']
        n_augment = int(data['n_augment'])
        noise_type = str(data['noise_type'])
        noise_level = float(data['noise_level'])

    print(f"Loaded augmented embeddings:")
    print(f"  Visual shape: {visual.shape}")
    print(f"  Semantic shape: {semantic.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Noise type: {noise_type}")
    print(f"  Noise level: {noise_level}")
    print(f"  Augmentations per sample: {n_augment}")

    # Concatenate visual and semantic embeddings
    combined = np.concatenate([visual, semantic], axis=1)  # (n_samples, 1536)
    print(f"\nCombined embeddings shape: {combined.shape}")

    # Apply t-SNE
    print(f"\nApplying t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(combined)
    print("t-SNE completed")

    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get unique labels
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    # --- Plot 1: All embeddings colored by class ---
    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=f'Class {label}', alpha=0.6, s=10)

    ax.set_title(f't-SNE Visualization (All Samples)\nNoise: {noise_type}, Level: {noise_level}', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', ncol=4, fontsize=6, markerscale=0.5)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Distinguish original vs augmented ---
    ax = axes[1]

    # Identify original samples (every (n_augment + 1)-th sample)
    n_words = len(labels) // (n_augment + 1)
    original_indices = [i * (n_augment + 1) for i in range(n_words)]
    augmented_indices = [i for i in range(len(labels)) if i not in original_indices]

    # Plot augmented samples (small, transparent)
    ax.scatter(embeddings_2d[augmented_indices, 0],
               embeddings_2d[augmented_indices, 1],
               c='lightblue', alpha=0.3, s=5, label=f'Augmented (n={len(augmented_indices)})')

    # Plot original samples (larger, opaque)
    ax.scatter(embeddings_2d[original_indices, 0],
               embeddings_2d[original_indices, 1],
               c='darkblue', alpha=0.8, s=50, edgecolors='black', linewidths=0.5,
               label=f'Original (n={len(original_indices)})')

    ax.set_title(f't-SNE Visualization (Original vs Augmented)\n{n_augment} augmented samples per original',
                 fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def visualize_noise_distribution(embeddings_path, output_path, n_samples=5):
    """
    Visualize the distribution of noise added to embeddings.

    Args:
        embeddings_path: Path to augmented_embeddings.npz file
        output_path: Path to save the output figure
        n_samples: Number of sample words to show
    """
    # Load augmented embeddings
    data = np.load(embeddings_path)
    visual = data['visual']
    labels = data['labels']
    n_augment = int(data['n_augment'])
    noise_type = str(data['noise_type'])
    noise_level = float(data['noise_level'])

    n_words = len(labels) // (n_augment + 1)

    # Select random words to visualize
    selected_word_ids = np.random.choice(n_words, size=min(n_samples, n_words), replace=False)

    fig, axes = plt.subplots(len(selected_word_ids), 1, figsize=(12, 3 * len(selected_word_ids)))
    if len(selected_word_ids) == 1:
        axes = [axes]

    for i, word_id in enumerate(selected_word_ids):
        ax = axes[i]

        # Get original embedding
        original_idx = word_id * (n_augment + 1)
        original_emb = visual[original_idx]

        # Get augmented embeddings for this word
        augmented_indices = [original_idx + j + 1 for j in range(n_augment)]
        augmented_embs = visual[augmented_indices]

        # Calculate noise (difference from original)
        noise = augmented_embs - original_emb  # (n_augment, d_visual)
        noise_flat = noise.flatten()

        # Plot histogram
        ax.hist(noise_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.set_title(f'Noise Distribution for Word {word_id} (Label: {labels[original_idx]})', fontsize=12)
        ax.set_xlabel('Noise Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_noise = np.mean(noise_flat)
        std_noise = np.std(noise_flat)
        ax.text(0.02, 0.98, f'Mean: {mean_noise:.4f}\nStd: {std_noise:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Noise Distribution Analysis\nNoise type: {noise_type}, Level: {noise_level}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Noise distribution visualization saved to: {output_path}")


def visualize_distance_statistics(embeddings_path, output_path):
    """
    Visualize L2 distance statistics between original and augmented embeddings.

    Args:
        embeddings_path: Path to augmented_embeddings.npz file
        output_path: Path to save the output figure
    """
    # Load augmented embeddings
    data = np.load(embeddings_path)
    visual = data['visual']
    semantic = data['semantic']
    labels = data['labels']
    n_augment = int(data['n_augment'])
    noise_type = str(data['noise_type'])
    noise_level = float(data['noise_level'])

    n_words = len(labels) // (n_augment + 1)

    # Calculate distances for visual and semantic embeddings
    visual_distances = []
    semantic_distances = []

    for word_id in range(n_words):
        original_idx = word_id * (n_augment + 1)
        original_visual = visual[original_idx]
        original_semantic = semantic[original_idx]

        for j in range(n_augment):
            aug_idx = original_idx + j + 1
            aug_visual = visual[aug_idx]
            aug_semantic = semantic[aug_idx]

            # Calculate L2 distances
            visual_dist = np.linalg.norm(aug_visual - original_visual)
            semantic_dist = np.linalg.norm(aug_semantic - original_semantic)

            visual_distances.append(visual_dist)
            semantic_distances.append(semantic_dist)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Visual distances
    ax = axes[0]
    ax.hist(visual_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(visual_distances), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(visual_distances):.4f}')
    ax.set_title('L2 Distance: Original vs Augmented (Visual)', fontsize=12)
    ax.set_xlabel('L2 Distance', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Semantic distances
    ax = axes[1]
    ax.hist(semantic_distances, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(semantic_distances), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(semantic_distances):.4f}')
    ax.set_title('L2 Distance: Original vs Augmented (Semantic)', fontsize=12)
    ax.set_xlabel('L2 Distance', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Distance Statistics\nNoise type: {noise_type}, Level: {noise_level}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distance statistics visualization saved to: {output_path}")
    print(f"  Visual - Mean: {np.mean(visual_distances):.4f}, Std: {np.std(visual_distances):.4f}")
    print(f"  Semantic - Mean: {np.mean(semantic_distances):.4f}, Std: {np.std(semantic_distances):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize augmented embeddings')
    parser.add_argument('embeddings_path', type=str,
                        help='Path to augmented_embeddings.npz file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='Number of t-SNE iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for t-SNE')
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualization types')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.embeddings_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate t-SNE visualization
    output_path = os.path.join(args.output_dir, 'tsne_visualization.png')
    visualize_tsne(args.embeddings_path, output_path, args.perplexity, args.n_iter, args.seed)

    if args.all:
        # Generate noise distribution visualization
        output_path = os.path.join(args.output_dir, 'noise_distribution.png')
        visualize_noise_distribution(args.embeddings_path, output_path)

        # Generate distance statistics visualization
        output_path = os.path.join(args.output_dir, 'distance_statistics.png')
        visualize_distance_statistics(args.embeddings_path, output_path)
