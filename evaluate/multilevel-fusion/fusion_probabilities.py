import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append('/mnt/afs/250010218/multi-level-Chinese-decoding/evaluate')
from embeddings_alignment.semantic_eval import evaluate_semantic_mapping
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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


def compute_similarities_and_probabilities(test_embeddings, GT_embeddings, temperature=1.0):
    """
    Compute cosine similarities and probabilities for each test sample

    Following the research algorithm in semantic_eval.py:
    1. Normalize embeddings (L2 normalization)
    2. Compute cosine similarities via matrix multiplication
    3. Use similarities directly for nearest prototype matching
    4. Optionally convert to probabilities via softmax for fusion

    Args:
        test_embeddings: (N, D) array of test embeddings
        GT_embeddings: (C, D) array of GT prototype embeddings
        temperature: Temperature for softmax (default=1.0)

    Returns:
        similarities: (N, C) array of cosine similarities (used for prediction)
        probabilities: (N, C) array of softmax probabilities (used for fusion)
    """
    # Normalize embeddings (following semantic_eval.py line 125-126)
    test_norm = normalize_embeddings(test_embeddings)
    GT_norm = normalize_embeddings(GT_embeddings)

    # Compute cosine similarities: (N, C) (following semantic_eval.py line 127)
    similarities = test_norm @ GT_norm.T

    # Convert similarities to probabilities via softmax for fusion
    # Note: predictions should use argmax(similarities), not argmax(probabilities)
    logits = similarities / temperature
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return similarities, probabilities


if __name__ == "__main__":
    print("="*80)
    print("Multi-Level Fusion: Computing Probabilities")
    print("="*80)

    # Load GT embeddings
    print("\n[1] Loading GT embeddings...")
    words_semantic, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed = \
        load_npz_semantic(GT_semantic_path)
    words_visual, emb_visual, _ = load_npz_vis(GT_visual_path)

    GT_semantic = emb_mean  # Using mean pooling for semantic
    GT_visual = emb_visual

    print(f"    - GT Semantic: {GT_semantic.shape} (words: {len(words_semantic)})")
    print(f"    - GT Visual: {GT_visual.shape} (chars: {len(words_visual)})")

    # Load test embeddings
    print("\n[2] Loading test embeddings...")
    labels_visual, embeddings_visual = load_test_embeddings(visual_path)
    labels_semantic, embeddings_semantic = load_test_embeddings(semantic_path)
    print(f"    - Test Visual: {embeddings_visual.shape}")
    print(f"    - Test Semantic: {embeddings_semantic.shape}")
    print(f"    - Number of test samples: {len(labels_visual)}")

    # Compute similarities and probabilities for visual embeddings
    print("\n[3] Computing similarities and probabilities for visual embeddings...")
    sims_visual, probs_visual = compute_similarities_and_probabilities(embeddings_visual, GT_visual)
    print(f"    - Visual similarities shape: {sims_visual.shape}")
    print(f"    - Visual probabilities shape: {probs_visual.shape}")
    print(f"    - Visual max similarity (first sample): {sims_visual[0].max():.6f}")
    print(f"    - Visual max probability (first sample): {probs_visual[0].max():.6f}")

    # Compute similarities and probabilities for semantic embeddings
    print("\n[4] Computing similarities and probabilities for semantic embeddings...")
    sims_semantic, probs_semantic = compute_similarities_and_probabilities(embeddings_semantic, GT_semantic)
    print(f"    - Semantic similarities shape: {sims_semantic.shape}")
    print(f"    - Semantic probabilities shape: {probs_semantic.shape}")
    print(f"    - Semantic max similarity (first sample): {sims_semantic[0].max():.6f}")
    print(f"    - Semantic max probability (first sample): {probs_semantic[0].max():.6f}")

    # Compute predictions using similarities (following semantic_eval.py line 134)
    print("\n[5] Computing predictions...")
    pred_visual = np.argmax(sims_visual, axis=1)  # Use similarities, not probabilities
    pred_semantic = np.argmax(sims_semantic, axis=1)  # Use similarities, not probabilities

    # Convert true labels to indices using GT vocabulary (following semantic_eval.py line 130-131)
    # IMPORTANT: Use GT label order for vocabulary, NOT label_list order!
    vocab_visual = {lab: i for i, lab in enumerate(words_visual)}
    vocab_semantic = {lab: i for i, lab in enumerate(words_semantic)}

    # Map test labels to GT indices for visual
    true_indices_visual = np.array([vocab_visual[lab] for lab in labels_visual])
    # Map test labels to GT indices for semantic
    true_indices_semantic = np.array([vocab_semantic[lab] for lab in labels_semantic])

    # Compute accuracies
    acc_visual = np.mean(pred_visual == true_indices_visual)
    acc_semantic = np.mean(pred_semantic == true_indices_semantic)

    print(f"    - Visual accuracy: {acc_visual:.4f}")
    print(f"    - Semantic accuracy: {acc_semantic:.4f}")

    # Fusion strategies
    # NOTE: For fusion, we assume visual and semantic use the same GT vocabulary
    # (which they do based on the GT embeddings)
    print("\n[6] Multi-level fusion...")
    true_indices = true_indices_visual  # Use visual GT vocab for fusion (same as semantic)

    # Strategy 1: Average similarities (following research algorithm)
    sims_avg = (sims_visual + sims_semantic) / 2
    pred_sims_avg = np.argmax(sims_avg, axis=1)
    acc_sims_avg = np.mean(pred_sims_avg == true_indices)
    print(f"    - Average similarity fusion accuracy: {acc_sims_avg:.4f}")

    # Strategy 2: Weighted similarity fusion (can be tuned)
    weight_visual = 0.3
    weight_semantic = 0.7
    sims_weighted = weight_visual * sims_visual + weight_semantic * sims_semantic
    pred_sims_weighted = np.argmax(sims_weighted, axis=1)
    acc_sims_weighted = np.mean(pred_sims_weighted == true_indices)
    print(f"    - Weighted similarity fusion accuracy (w_v={weight_visual}, w_s={weight_semantic}): {acc_sims_weighted:.4f}")

    # Strategy 3: Max similarity fusion
    sims_max = np.maximum(sims_visual, sims_semantic)
    pred_sims_max = np.argmax(sims_max, axis=1)
    acc_sims_max = np.mean(pred_sims_max == true_indices)
    print(f"    - Max similarity fusion accuracy: {acc_sims_max:.4f}")

    # Strategy 4: Average probabilities (alternative fusion method)
    probs_avg = (probs_visual + probs_semantic) / 2
    pred_probs_avg = np.argmax(probs_avg, axis=1)
    acc_probs_avg = np.mean(pred_probs_avg == true_indices)
    print(f"    - Average probability fusion accuracy: {acc_probs_avg:.4f}")

    # Strategy 5: Weighted probability fusion
    probs_weighted = weight_visual * probs_visual + weight_semantic * probs_semantic
    pred_probs_weighted = np.argmax(probs_weighted, axis=1)
    acc_probs_weighted = np.mean(pred_probs_weighted == true_indices)
    print(f"    - Weighted probability fusion accuracy: {acc_probs_weighted:.4f}")

