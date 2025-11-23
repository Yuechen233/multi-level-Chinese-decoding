#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Level Fusion: Concatenated Cosine Similarity Evaluation

Evaluates test embeddings using concatenated visual+semantic embeddings
with double normalization approach matching the training procedure.
"""

import numpy as np
import sys
sys.path.append('/mnt/afs/250010218/multi-level-Chinese-decoding/evaluate')

# Paths configuration
visual_path='evaluate/multilevel-fusion/sub11/visual_embeddings_epoch_300.npy'
semantic_path='evaluate/multilevel-fusion/sub11/semantic_embeddings_epoch_300.npy'

# GT embeddings paths
GT_semantic_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Semantic_GT_bert.npz'
GT_visual_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Visual_GT_VitPerchar.npz'

# Label list (61 words)
label_list = ['丝瓜', '你', '关门', '凳子', '厕所', '口渴', '吃',
               '喝', '嘴巴', '外卖', '头疼', '家人', '小刀', '帮助',
                  '平静', '心情', '怎样', '感觉', '愿意', '我', '手机',
                    '找', '把', '护士', '拿', '换药', '放在', '是', '有',
                          '朋友', '橙汁', '毛巾', '汤圆', '漂亮', '热水',
                            '猪肉', '玩', '电脑', '看', '碗', '穿', '篮球',
                                  '米饭', '给', '脸盆', '菠萝', '葱花', '蒜泥',
                                      '衣服', '豆腐', '软糖', '醋', '钢琴', '问题',
                                          '需要', '青菜', '面条', '音乐', '预约', '香肠', '鱼块']


# Helper functions
def load_npz_vis(path):
    """Load visual embeddings from .npz file"""
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"], dict(data["meta"])


def load_npz_semantic(path):
    """Load semantic embeddings from .npz file"""
    data = np.load(path, allow_pickle=True)
    words = data['words']
    emb_cls = data['emb_cls']
    emb_mean = data['emb_mean']
    emb_max = data['emb_max']
    emb_weighted = data['emb_weighted']
    emb_mixed = data['emb_mixed']
    return words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed


def load_test_embeddings(path):
    """Load test embeddings from .npy file"""
    data = np.load(path, allow_pickle=True)
    labels = data[:, -1]
    # Convert label elements to int
    labels = labels.astype(int)
    labels = [label_list[i] for i in labels]
    labels = np.array(labels)
    embeddings = data[:, 0:-1]
    return labels, embeddings


def normalize_embeddings(embeddings):
    """L2 normalize embeddings"""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def create_concat_embeddings(visual_emb, semantic_emb):
    """
    Create concatenated embeddings with double normalization.

    Following the approach in embddings_cls_pretrain/train_emb_cls.py:264-273:
    1. Normalize visual embeddings → unit vectors
    2. Normalize semantic embeddings → unit vectors
    3. Concatenate normalized embeddings
    4. Re-normalize concatenated result → unit vector

    Args:
        visual_emb: (N, d_visual) array
        semantic_emb: (N, d_semantic) array

    Returns:
        concat_norm: (N, d_visual + d_semantic) array of unit vectors
    """
    # Step 1-2: Normalize each modality separately
    visual_norm = normalize_embeddings(visual_emb)
    semantic_norm = normalize_embeddings(semantic_emb)

    # Step 3: Concatenate after normalization
    concat = np.concatenate([visual_norm, semantic_norm], axis=1)

    # Step 4: Normalize the concatenated result again
    concat_norm = normalize_embeddings(concat)

    return concat_norm




if __name__ == "__main__":
    print("="*80)
    print("Multi-Level Fusion: Concatenated Cosine Similarity")
    print("="*80)

    # Load GT embeddings
    print("\n[1] Loading GT embeddings...")
    words_semantic, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed = \
        load_npz_semantic(GT_semantic_path)
    words_visual, emb_visual, _ = load_npz_vis(GT_visual_path)

    GT_semantic = emb_mean  # Using mean pooling for semantic
    GT_visual = emb_visual

    print(f"    - GT Visual: {GT_visual.shape} (chars: {len(words_visual)})")
    print(f"    - GT Semantic: {GT_semantic.shape} (words: {len(words_semantic)})")

    # Load test embeddings
    print("\n[2] Loading test embeddings...")
    labels_visual, embeddings_visual = load_test_embeddings(visual_path)
    labels_semantic, embeddings_semantic = load_test_embeddings(semantic_path)
    print(f"    - Test Visual: {embeddings_visual.shape}")
    print(f"    - Test Semantic: {embeddings_semantic.shape}")
    print(f"    - Number of test samples: {len(labels_visual)}")

    # Create concatenated embeddings with double normalization
    print("\n[3] Creating concatenated embeddings with double normalization...")
    GT_concat = create_concat_embeddings(GT_visual, GT_semantic)
    test_concat = create_concat_embeddings(embeddings_visual, embeddings_semantic)
    print(f"    - GT Concat: {GT_concat.shape}")
    print(f"    - Test Concat: {test_concat.shape}")

    # Compute cosine similarity
    print("\n[4] Computing cosine similarity accuracy...")
    # Cosine similarity via matrix multiplication (embeddings already normalized)
    similarities = test_concat @ GT_concat.T  # (N, C)
    print(f"    - Similarity matrix shape: {similarities.shape}")
    print(f"    - Max similarity (first sample): {similarities[0].max():.6f}")

    # Predict using argmax
    predictions = np.argmax(similarities, axis=1)

    # Map true labels to GT vocabulary indices
    # IMPORTANT: Use GT label order for vocabulary, NOT label_list order!
    vocab = {lab: i for i, lab in enumerate(words_visual)}
    true_indices = np.array([vocab[lab] for lab in labels_visual])

    # Compute accuracy
    accuracy = np.mean(predictions == true_indices)

    print(f"\n[5] Results:")
    print(f"    - Concatenated embeddings accuracy: {accuracy:.4f}")

    # Additional statistics
    print(f"\n[6] Additional Statistics:")
    print(f"    - Mean similarity (all predictions): {similarities.max(axis=1).mean():.6f}")
    print(f"    - Std similarity (all predictions): {similarities.max(axis=1).std():.6f}")
    print(f"    - Min max similarity: {similarities.max(axis=1).min():.6f}")
    print(f"    - Max max similarity: {similarities.max(axis=1).max():.6f}")

    print("\n" + "="*80)
    print("Evaluation Complete")
    print("="*80)

