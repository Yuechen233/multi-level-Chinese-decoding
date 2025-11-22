#!/usr/bin/env python3
"""
Joint multi-task + end-to-end classification training script for Du-IN model.
Combines semantic, visual, and acoustic alignment tasks with end-to-end 61-word classification.
All four losses (semantic + visual + acoustic + e2e classification) are trained jointly.

Created for joint multi-task + e2e classification implementation
@author: Claude Code
"""
import matplotlib.pyplot as plt
import torch
import os, time
import argparse
import copy as cp
import numpy as np
import scipy as sp
from collections import Counter
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import load_pickle
from models.duin import duin_joint_multitask_cls as duin_model

# GPU DEBUGGING: Disable cuDNN to test if it causes GPU training failure
import torch.backends.cudnn as cudnn
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True
print("WARNING: cuDNN DISABLED for debugging GPU training issue")

__all__ = [
    "init",
    "train",
]

# Global variables.
params = None; paths = None
model = None; optimizer = None

"""
init funcs
"""
# def init func
def init(params_):
    """
    Initialize `duin_multitask` training variables.

    Args:
        params_: DotDict - The parameters of current training process.

    Returns:
        None
    """
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)
    paths = utils.Paths(base=params.train.base, params=params)
    paths.run.logger.tensorboard = SummaryWriter(paths.run.train)
    # Initialize model & training.
    _init_model(); _init_train()
    # Log the completion of initialization.
    msg = (
        "INFO: Complete the initialization of the training process with params ({})."
    ).format(params); print(msg); paths.run.logger.summaries.info(msg)

    # --- Add missing folders ---
    paths.run.ckpt = os.path.join(paths.run.train, "ckpt")
    os.makedirs(paths.run.ckpt, exist_ok=True)
    print(f"[INFO] Checkpoint directory created at: {paths.run.ckpt}")

    paths.run.save_embeddings = os.path.join(paths.run.train, "save_embeddings")
    os.makedirs(paths.run.save_embeddings, exist_ok=True)
    print(f"[INFO] Embeddings directory created at: {paths.run.save_embeddings}")

    paths.run.summaries = os.path.join(paths.run.train, "summaries")
    os.makedirs(paths.run.summaries, exist_ok=True)
    print(f"[INFO] Summaries directory created at: {paths.run.summaries}")

    # --- Save training metadata (in same directory as summaries.log) ---
    training_info_path = os.path.join(paths.run.base, "training_info.txt")
    subj_i = params.train.subjs[0]
    with open(training_info_path, "w") as f:
        f.write(f"Training Type: joint_multitask_cls (semantic + visual + acoustic + e2e classification)\n")
        f.write(f"Subject Number: {subj_i}\n")
        f.write(f"Task Weights: semantic={params.model.task_weight_semantic}, visual={params.model.task_weight_visual}, acoustic={params.model.task_weight_acoustic}, e2e={params.model.task_weight_e2e}\n")
        f.write(f"Uncertainty Weighting: {params.model.use_uncertainty_weighting}\n")
        f.write(f"Acoustic Contrastive: {params.model.acoustic_use_contra}\n")
    print(f"[INFO] Training metadata saved to: {training_info_path}")

    # --- Save shell script if provided ---
    if hasattr(params.train, 'run_script') and params.train.run_script is not None:
        import shutil
        script_name = os.path.basename(params.train.run_script)
        script_dest = os.path.join(paths.run.script, script_name)
        try:
            shutil.copy2(params.train.run_script, script_dest)
            msg = f"[INFO] Shell script saved to: {script_dest}"
            print(msg); paths.run.logger.summaries.info(msg)
        except Exception as e:
            msg = f"[WARNING] Failed to save shell script: {e}"
            print(msg); paths.run.logger.summaries.warning(msg)

# def _init_model func
def _init_model():
    """
    Initialize model used in the training process.

    Args:
        None

    Returns:
        None
    """
    global params
    # Initialize torch configuration.
    torch.set_default_dtype(getattr(torch, params._precision))
    # Set the internal precision of float32 matrix multiplications.
    torch.set_float32_matmul_precision("high")

# def _init_train func
def _init_train():
    """
    Initialize the training process.

    Args:
        None

    Returns:
        None
    """
    pass

"""
data funcs
"""
# def load_data func
def load_data(load_params):
    """
    Load data from specified dataset.

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset.
        dataset_validation: torch.utils.data.DataLoader - The validation dataset.
        dataset_test: torch.utils.data.DataLoader - The test dataset.
    """
    global params
    # Load data from specified dataset.
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train, dataset_validation, dataset_test = func(load_params)
    except AttributeError:
        raise ValueError((
            "ERROR: Unknown dataset type {} in train.duin.run_multitask."
        ).format(params.train.dataset))
    # Return the final datasets.
    return dataset_train, dataset_validation, dataset_test

# def _load_data_seeg_he2023xuanwu func
def _load_data_seeg_he2023xuanwu(load_params):
    """
    Load seeg data from the specified subject in `seeg_he2023xuanwu`.
    This version loads ALL THREE target types (semantic + visual + acoustic).

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset.
        dataset_validation: torch.utils.data.DataLoader - The validation dataset.
        dataset_test: torch.utils.data.DataLoader - The test dataset.
    """
    global params, paths

    # === Load all three GT embedding/label tables ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # Load semantic GT (BERT embeddings)
    semantic_path = os.path.join(base_dir, "GT_embeddings/Duin_Semantic_GT_bert.npz")
    semantic_data = np.load(semantic_path, allow_pickle=True)
    semantic_table = semantic_data["emb_mean"]  # Use emb_mean for semantic
    semantic_words = semantic_data["words"]     # Use words for semantic
    semantic_table = semantic_table / (np.linalg.norm(semantic_table, axis=1, keepdims=True) + 1e-8)
    print(f"[INFO] Loaded semantic GT: {semantic_table.shape}")

    # Load visual GT (ViT embeddings)
    visual_path = os.path.join(base_dir, "GT_embeddings/Duin_Visual_GT_VitPerchar.npz")
    visual_data = np.load(visual_path, allow_pickle=True)
    visual_table = visual_data["embeddings"]    # Use embeddings for visual
    visual_words = visual_data["chars"]         # Use chars for visual
    visual_table = visual_table / (np.linalg.norm(visual_table, axis=1, keepdims=True) + 1e-8)
    print(f"[INFO] Loaded visual GT: {visual_table.shape}")

    # Load acoustic GT (tone labels)
    acoustic_path = os.path.join(base_dir, "GT_embeddings/Duin_Acoustic_label.npz")
    acoustic_data = np.load(acoustic_path, allow_pickle=True)
    acoustic_words = acoustic_data["chars"]      # Use chars for acoustic
    acoustic_labels = acoustic_data["embeddings"] # Use embeddings (contains tone labels)
    print(f"[INFO] Loaded acoustic GT labels: {acoustic_labels.shape}")

    # === Collect label order from dataset (same as run_align_vis.py) ===
    import unicodedata
    def _norm_text(s):
        return unicodedata.normalize("NFKC", str(s)).strip()

    subjs_cfg = load_params.subjs_cfg
    label_name_set = set()
    subj0_path = subjs_cfg[0].path
    task_dir = os.path.join(subj0_path, "word-recitation")
    dataset_dir_name = f"dataset.bipolar.default.{'aligned' if load_params.use_align else 'unaligned'}"
    for run_i in os.listdir(task_dir):
        data_pkl = os.path.join(task_dir, run_i, dataset_dir_name, "data")
        if not os.path.exists(data_pkl): continue
        dataset_data_i = load_pickle(data_pkl)
        for sample in dataset_data_i:
            if hasattr(sample, "name"):
                label_name_set.add(_norm_text(sample.name))
    label_order = sorted(label_name_set)
    print(f"[INFO] Dataset has {len(label_order)} unique labels")

    # === Align all three tables to label_order ===
    def align_table(table, words):
        words_norm = [_norm_text(w) for w in words]
        word_to_idx = {w: i for i, w in enumerate(words_norm)}
        aligned = np.zeros((len(label_order), table.shape[1]), dtype=table.dtype)
        missing = []
        for i, name in enumerate(label_order):
            j = word_to_idx.get(_norm_text(name), None)
            if j is not None:
                aligned[i, :] = table[j, :]
            else:
                missing.append(name)
        if missing:
            print(f"[WARNING] {len(missing)} labels not found in embedding table")
        return aligned

    semantic_table_aligned = align_table(semantic_table, semantic_words)
    visual_table_aligned = align_table(visual_table, visual_words)

    # Align acoustic labels (tone labels are in acoustic_labels as [tone1, tone2])
    def align_acoustic_labels(labels, words):
        """Align acoustic tone labels to label_order"""
        words_norm = [_norm_text(w) for w in words]
        word_to_idx = {w: i for i, w in enumerate(words_norm)}
        # Initialize aligned labels (shape: [n_labels, 2] for tone1 and tone2)
        aligned = np.zeros((len(label_order), 2), dtype=np.int64)
        missing = []
        for i, name in enumerate(label_order):
            j = word_to_idx.get(_norm_text(name), None)
            if j is not None:
                aligned[i, :] = labels[j, :]  # [tone1, tone2]
            else:
                missing.append(name)
        if missing:
            print(f"[WARNING] {len(missing)} labels not found in acoustic label table")
        return aligned

    acoustic_labels_aligned = align_acoustic_labels(acoustic_labels, acoustic_words)
    acoustic_tone1_aligned = acoustic_labels_aligned[:, 0] - 1  # Convert to 0-indexed (1-5 -> 0-4)
    acoustic_tone2_aligned = acoustic_labels_aligned[:, 1] - 1  # Convert to 0-indexed (1-5 -> 0-4)

    # === Initialize configuration (same as existing scripts) ===
    n_channels = load_params.n_channels if load_params.n_channels is not None else None
    n_subjects = load_params.n_subjects if load_params.n_subjects is not None else len(subjs_cfg)
    subj_idxs = load_params.subj_idxs if load_params.subj_idxs is not None else np.arange(n_subjects)
    seq_len = None; n_labels = None

    # === Initialize data containers ===
    Xs_train = []; ys_semantic_train = []; ys_visual_train = []
    ys_acoustic_tone1_train = []; ys_acoustic_tone2_train = []; subj_ids_train = []
    y_idx_train = []  # ← Add integer label storage for train
    Xs_validation = []; ys_semantic_validation = []; ys_visual_validation = []
    ys_acoustic_tone1_validation = []; ys_acoustic_tone2_validation = []; subj_ids_validation = []
    y_idx_validation = []  # ← Add integer label storage for validation
    Xs_test = []; ys_semantic_test = []; ys_visual_test = []
    ys_acoustic_tone1_test = []; ys_acoustic_tone2_test = []; subj_ids_test = []
    y_idx_test = []  # ← Add integer label storage for test

    # === Loop through subjects (same as existing scripts) ===
    for subj_idx, subj_cfg_i in zip(subj_idxs, subjs_cfg):
        # Load data from specified subject
        func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
        dataset = func(subj_cfg_i.path, ch_names=subj_cfg_i.ch_names, use_align=load_params.use_align)
        X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)

        # Preprocess data (same as run_align_vis.py)
        if load_params.type.startswith("bipolar"):
            sample_rate = 1000
            X = sp.signal.resample(X, int(np.round(X.shape[1] / (sample_rate / load_params.resample_rate))), axis=1)
            X = X[:,int(np.round((0.0 - (-0.5)) * load_params.resample_rate)):
                   int(np.round((2.5 - (-0.5)) * load_params.resample_rate)),:]
            X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
        else:
            raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))

        # Train/test split (same as run_align_vis.py)
        train_ratio = params.train.train_ratio
        train_idxs = []; test_idxs = []
        for label_i in sorted(set(y)):
            label_idxs = np.where(y == label_i)[0].tolist()
            train_idxs.extend(label_idxs[:int(train_ratio * len(label_idxs))])
            test_idxs.extend(label_idxs[int(train_ratio * len(label_idxs)):])
        for train_idx in train_idxs: assert train_idx not in test_idxs
        train_idxs = np.array(train_idxs, dtype=np.int64); test_idxs = np.array(test_idxs, dtype=np.int64)
        X_train = X[train_idxs,:,:]; y_train = y[train_idxs]
        X_test = X[test_idxs,:,:]; y_test = y[test_idxs]

        if len(X_train) == 0 or len(X_test) == 0: continue

        # Get sorted labels
        labels = sorted(set(y_train))
        assert len(set(y_train)) == len(set(y_test))
        y_train_idx = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64)
        y_test_idx = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64)

        # === Map to ALL THREE targets ===
        # Semantic embeddings
        y_semantic_train = semantic_table_aligned[y_train_idx]
        y_semantic_test = semantic_table_aligned[y_test_idx]

        # Visual embeddings
        y_visual_train = visual_table_aligned[y_train_idx]
        y_visual_test = visual_table_aligned[y_test_idx]

        # Acoustic tone labels (convert to one-hot)
        y_tone1_train = acoustic_tone1_aligned[y_train_idx]
        y_tone1_test = acoustic_tone1_aligned[y_test_idx]
        y_tone2_train = acoustic_tone2_aligned[y_train_idx]
        y_tone2_test = acoustic_tone2_aligned[y_test_idx]
        y_tone1_train_oh = np.eye(5)[y_tone1_train]
        y_tone1_test_oh = np.eye(5)[y_tone1_test]
        y_tone2_train_oh = np.eye(5)[y_tone2_train]
        y_tone2_test_oh = np.eye(5)[y_tone2_test]

        # Validation/test split (same as run_align_vis.py)
        validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
        validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
        X_validation = X_test[validation_mask,:,:]; X_test = X_test[~validation_mask,:,:]
        y_semantic_validation = y_semantic_test[validation_mask,:]; y_semantic_test = y_semantic_test[~validation_mask,:]
        y_visual_validation = y_visual_test[validation_mask,:]; y_visual_test = y_visual_test[~validation_mask,:]
        y_tone1_validation = y_tone1_test_oh[validation_mask,:]; y_tone1_test = y_tone1_test_oh[~validation_mask,:]
        y_tone2_validation = y_tone2_test_oh[validation_mask,:]; y_tone2_test = y_tone2_test_oh[~validation_mask,:]
        # Split integer labels too (use different variable names to avoid collision)
        y_idx_validation_i = y_test_idx[validation_mask]; y_idx_test_i = y_test_idx[~validation_mask]

        # Create subject IDs (same as run_align_vis.py)
        subj_id_train = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_train.shape[0])])
        subj_id_validation = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_validation.shape[0])])
        subj_id_test = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_test.shape[0])])

        # Log information
        msg = (
            "INFO: Data preparation for subject ({}) complete, with train-set ({}) & validation-set ({}) & test-set ({})."
        ).format(subj_cfg_i.name, X_train.shape, X_validation.shape, X_test.shape)
        print(msg); paths.run.logger.summaries.info(msg)

        # Append to lists
        Xs_train.append(X_train); ys_semantic_train.append(y_semantic_train); ys_visual_train.append(y_visual_train)
        ys_acoustic_tone1_train.append(y_tone1_train_oh); ys_acoustic_tone2_train.append(y_tone2_train_oh)
        subj_ids_train.append(subj_id_train)
        y_idx_train.append(y_train_idx)  # ← Store integer labels

        Xs_validation.append(X_validation); ys_semantic_validation.append(y_semantic_validation)
        ys_visual_validation.append(y_visual_validation)
        ys_acoustic_tone1_validation.append(y_tone1_validation); ys_acoustic_tone2_validation.append(y_tone2_validation)
        subj_ids_validation.append(subj_id_validation)
        y_idx_validation.append(y_idx_validation_i)  # ← Store integer labels (use _i suffix)

        Xs_test.append(X_test); ys_semantic_test.append(y_semantic_test); ys_visual_test.append(y_visual_test)
        ys_acoustic_tone1_test.append(y_tone1_test); ys_acoustic_tone2_test.append(y_tone2_test)
        subj_ids_test.append(subj_id_test)
        y_idx_test.append(y_idx_test_i)  # ← Store integer labels (use _i suffix)

        # Update params
        n_channels = max(X.shape[-1], n_channels) if n_channels is not None else X.shape[-1]
        seq_len = X.shape[-2] if seq_len is None else seq_len; assert seq_len == X.shape[-2]
        n_labels = len(labels) if n_labels is None else n_labels; assert n_labels == len(labels)

    # === Pad channels to max (same as run_align_vis.py) ===
    if load_params.n_channels is not None: assert n_channels == load_params.n_channels
    Xs_train = [np.concatenate([X_i, np.zeros((*X_i.shape[:-1], n_channels - X_i.shape[-1]), dtype=X_i.dtype)], axis=-1)
                for X_i in Xs_train]
    Xs_validation = [np.concatenate([X_i, np.zeros((*X_i.shape[:-1], n_channels - X_i.shape[-1]), dtype=X_i.dtype)], axis=-1)
                     for X_i in Xs_validation]
    Xs_test = [np.concatenate([X_i, np.zeros((*X_i.shape[:-1], n_channels - X_i.shape[-1]), dtype=X_i.dtype)], axis=-1)
               for X_i in Xs_test]

    # === Concatenate all subjects (same as run_align_vis.py) ===
    Xs_train = np.concatenate(Xs_train, axis=0)
    ys_semantic_train = np.concatenate(ys_semantic_train, axis=0)
    ys_visual_train = np.concatenate(ys_visual_train, axis=0)
    ys_acoustic_tone1_train = np.concatenate(ys_acoustic_tone1_train, axis=0)
    ys_acoustic_tone2_train = np.concatenate(ys_acoustic_tone2_train, axis=0)
    subj_ids_train = np.concatenate(subj_ids_train, axis=0)
    y_idx_train = np.concatenate(y_idx_train, axis=0)  # ← Concatenate integer labels

    Xs_validation = np.concatenate(Xs_validation, axis=0)
    ys_semantic_validation = np.concatenate(ys_semantic_validation, axis=0)
    ys_visual_validation = np.concatenate(ys_visual_validation, axis=0)
    ys_acoustic_tone1_validation = np.concatenate(ys_acoustic_tone1_validation, axis=0)
    ys_acoustic_tone2_validation = np.concatenate(ys_acoustic_tone2_validation, axis=0)
    subj_ids_validation = np.concatenate(subj_ids_validation, axis=0)
    y_idx_validation = np.concatenate(y_idx_validation, axis=0)  # ← Concatenate integer labels

    Xs_test = np.concatenate(Xs_test, axis=0)
    ys_semantic_test = np.concatenate(ys_semantic_test, axis=0)
    ys_visual_test = np.concatenate(ys_visual_test, axis=0)
    ys_acoustic_tone1_test = np.concatenate(ys_acoustic_tone1_test, axis=0)
    ys_acoustic_tone2_test = np.concatenate(ys_acoustic_tone2_test, axis=0)
    subj_ids_test = np.concatenate(subj_ids_test, axis=0)
    y_idx_test = np.concatenate(y_idx_test, axis=0)  # ← Concatenate integer labels

    # === Shuffle dataset (same as run_align_vis.py) ===
    train_idxs = np.arange(Xs_train.shape[0]); np.random.shuffle(train_idxs)
    validation_idxs = np.arange(Xs_validation.shape[0]); np.random.shuffle(validation_idxs)
    test_idxs = np.arange(Xs_test.shape[0]); np.random.shuffle(test_idxs)

    Xs_train = Xs_train[train_idxs,...]; ys_semantic_train = ys_semantic_train[train_idxs,...]
    ys_visual_train = ys_visual_train[train_idxs,...]; ys_acoustic_tone1_train = ys_acoustic_tone1_train[train_idxs,...]
    ys_acoustic_tone2_train = ys_acoustic_tone2_train[train_idxs,...]; subj_ids_train = subj_ids_train[train_idxs,...]
    y_idx_train = y_idx_train[train_idxs]  # ← Shuffle integer labels

    Xs_validation = Xs_validation[validation_idxs,...]; ys_semantic_validation = ys_semantic_validation[validation_idxs,...]
    ys_visual_validation = ys_visual_validation[validation_idxs,...]
    ys_acoustic_tone1_validation = ys_acoustic_tone1_validation[validation_idxs,...]
    ys_acoustic_tone2_validation = ys_acoustic_tone2_validation[validation_idxs,...]
    subj_ids_validation = subj_ids_validation[validation_idxs,...]
    y_idx_validation = y_idx_validation[validation_idxs]  # ← Shuffle integer labels

    Xs_test = Xs_test[test_idxs,...]; ys_semantic_test = ys_semantic_test[test_idxs,...]
    ys_visual_test = ys_visual_test[test_idxs,...]; ys_acoustic_tone1_test = ys_acoustic_tone1_test[test_idxs,...]
    ys_acoustic_tone2_test = ys_acoustic_tone2_test[test_idxs,...]; subj_ids_test = subj_ids_test[test_idxs,...]
    y_idx_test = y_idx_test[test_idxs]  # ← Shuffle integer labels

    # Log information
    msg = (
        "INFO: Data preparation complete, with train-set ({}) & validation-set ({}) & test-set ({})."
    ).format(Xs_train.shape, Xs_validation.shape, Xs_test.shape)
    print(msg); paths.run.logger.summaries.info(msg)

    # === Construct datasets (using custom MultitaskDataset class) ===
    dataset_train = MultitaskDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_semantic": ys_i, "y_visual": yv_i, "y_tone1": yt1_i, "y_tone2": yt2_i, "subj_id": subj_id_i, "y_idx": y_idx_i,
    }) for X_i, ys_i, yv_i, yt1_i, yt2_i, subj_id_i, y_idx_i in zip(
        Xs_train, ys_semantic_train, ys_visual_train, ys_acoustic_tone1_train, ys_acoustic_tone2_train, subj_ids_train, y_idx_train
    )], use_aug=True)

    dataset_validation = MultitaskDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_semantic": ys_i, "y_visual": yv_i, "y_tone1": yt1_i, "y_tone2": yt2_i, "subj_id": subj_id_i, "y_idx": y_idx_i,
    }) for X_i, ys_i, yv_i, yt1_i, yt2_i, subj_id_i, y_idx_i in zip(
        Xs_validation, ys_semantic_validation, ys_visual_validation, ys_acoustic_tone1_validation, ys_acoustic_tone2_validation, subj_ids_validation, y_idx_validation
    )], use_aug=False)

    dataset_test = MultitaskDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_semantic": ys_i, "y_visual": yv_i, "y_tone1": yt1_i, "y_tone2": yt2_i, "subj_id": subj_id_i, "y_idx": y_idx_i,
    }) for X_i, ys_i, yv_i, yt1_i, yt2_i, subj_id_i, y_idx_i in zip(
        Xs_test, ys_semantic_test, ys_visual_test, ys_acoustic_tone1_test, ys_acoustic_tone2_test, subj_ids_test, y_idx_test
    )], use_aug=False)

    # === Create dataloaders (same as run_align_vis.py) ===
    dataset_train = torch.utils.data.DataLoader(dataset_train,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_validation = torch.utils.data.DataLoader(dataset_validation,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)

    # === Update params (same as run_align_vis.py) ===
    params.model.subj.n_subjects = params.model.n_subjects = n_subjects
    params.model.subj.d_input = params.model.n_channels = n_channels
    assert seq_len % params.model.seg_len == 0; params.model.seq_len = seq_len
    token_len = params.model.seq_len // params.model.tokenizer.seg_len
    params.model.tokenizer.token_len = token_len
    params.model.encoder.emb_len = token_len

    params.model.semantic_align.d_feature = (
        params.model.encoder.d_model * params.model.encoder.emb_len
    )
    params.model.visual_align.d_feature = (
        params.model.encoder.d_model * params.model.encoder.emb_len
    )

    # Return the final datasets
    return dataset_train, dataset_validation, dataset_test

# def MultitaskDataset class
class MultitaskDataset(torch.utils.data.Dataset):
    """
    Multi-task dataset for semantic + visual + acoustic tasks.
    """

    def __init__(self, data_items, use_aug=False, **kwargs):
        """
        Initialize `MultitaskDataset` object.

        Args:
            data_items: list - The list of data items, including [X, y_semantic, y_visual, y_tone1, y_tone2, subj_id].
            use_aug: bool - The flag that indicates whether enable augmentations.
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `torch.utils.data.Dataset`
        # style model and inherit it's functionality.
        super(MultitaskDataset, self).__init__(**kwargs)

        # Initialize parameters.
        self.data_items = data_items; self.use_aug = use_aug

    # def __len__ func
    def __len__(self):
        """
        Get the number of samples of dataset.

        Args:
            None

        Returns:
            n_samples: int - The number of samples of dataset.
        """
        return len(self.data_items)

    # def __getitem__ func
    def __getitem__(self, index):
        """
        Get the data item at the specified index.

        Args:
            index: int - The index of the specified data item.

        Returns:
            X_i: (seq_len, n_channels) - The brain signal.
            y_semantic_i: (768,) - The semantic BERT embedding.
            y_visual_i: (768,) - The visual ViT embedding.
            y_tone1_i: (5,) - The first tone label (one-hot).
            y_tone2_i: (5,) - The second tone label (one-hot).
            subj_id_i: (n_subjects,) - The subject ID.
            y_idx_i: int - The integer class label (0-60).
        """
        # Get the data item at the specified index.
        data_item_i = cp.deepcopy(self.data_items[index])
        # Get the X, y, and subj_id from the data item.
        X_i = data_item_i.X; subj_id_i = data_item_i.subj_id
        y_semantic_i = data_item_i.y_semantic; y_visual_i = data_item_i.y_visual
        y_tone1_i = data_item_i.y_tone1; y_tone2_i = data_item_i.y_tone2
        y_idx_i = data_item_i.y_idx  # ← Get integer label

        # Data augmentation (same as run_align_vis.py)
        if self.use_aug:
            # Randomly select an augmentation function.
            aug_func_i = np.random.choice(["noise", "scale", "negate", "time-flip", "none"], p=[0.1,0.1,0.1,0.1,0.6])
            # Execute the selected augmentation function.
            if aug_func_i == "noise":
                X_i = X_i + 0.1 * np.std(X_i, axis=0, keepdims=True) * np.random.randn(*X_i.shape)
            elif aug_func_i == "scale":
                X_i = X_i * np.random.uniform(low=0.8, high=1.2)
            elif aug_func_i == "negate":
                X_i = -X_i
            elif aug_func_i == "time-flip":
                X_i = X_i[::-1,:].copy()

        # Return the final data item.
        # Note: X_i is stored as (n_channels, seq_len), transpose to (seq_len, n_channels) for model
        return (
            torch.from_numpy(X_i.T.astype(np.float32)),
            torch.from_numpy(y_semantic_i.astype(np.float32)),
            torch.from_numpy(y_visual_i.astype(np.float32)),
            torch.from_numpy(y_tone1_i.astype(np.float32)),
            torch.from_numpy(y_tone2_i.astype(np.float32)),
            torch.from_numpy(subj_id_i.astype(np.float32)),
            torch.tensor(y_idx_i, dtype=torch.int64),  # ← Return integer label
        )

"""
train funcs
"""
# def log_loss_block func
def log_loss_block(phase_name, loss_dict):
    """
    Format loss information for logging.

    Args:
        phase_name: str - The phase name (train, validation, test).
        loss_dict: dict - Dictionary of loss values.

    Returns:
        msg: str - Formatted loss message.
    """
    loss_keys = list(loss_dict.keys())
    msg = f"Loss({phase_name}): {loss_dict[loss_keys[0]]:.5f} ({loss_keys[0]})"
    for loss_idx in range(1, len(loss_keys)):
        msg += "; {:.5f} ({})".format(loss_dict[loss_keys[loss_idx]], loss_keys[loss_idx])
    return msg

# def _evaluate func
def _evaluate(model, dataloader, params, device):
    """
    Evaluate model on a dataset (validation or test).

    Args:
        model: torch.nn.Module - The model to evaluate.
        dataloader: DataLoader - The dataloader for evaluation.
        params: DotDict - The parameters.
        device: str - The device to use.

    Returns:
        loss_dict: DotDict - Dictionary of averaged losses.
        semantic_embeddings: np.ndarray - Predicted semantic embeddings.
        visual_embeddings: np.ndarray - Predicted visual embeddings.
        semantic_labels: np.ndarray - GT semantic embeddings (768-d).
        visual_labels: np.ndarray - GT visual embeddings (768-d).
        acoustic_results: dict - Acoustic classification results (accuracy, predictions).
        y_idx_all: np.ndarray - Integer class labels (0-60).
    """
    model.eval()
    loss_dict = utils.DotDict()

    # Collect embeddings and predictions
    all_semantic_embs = []
    all_visual_embs = []
    all_semantic_labels = []
    all_visual_labels = []
    all_tone1_preds = []
    all_tone1_labels = []
    all_tone2_preds = []
    all_tone2_labels = []
    all_y_idx = []  # ← Collect integer labels
    all_y_pred = []  # ← Collect e2e classification predictions (NEW)

    with torch.no_grad():
        for batch_data in dataloader:
            X, y_semantic, y_visual, y_tone1, y_tone2, subj_id, y_idx = batch_data  # ← Unpack y_idx
            X = X.to(device=device, dtype=torch.float32)
            y_semantic = y_semantic.to(device=device, dtype=torch.float32)
            y_visual = y_visual.to(device=device, dtype=torch.float32)
            y_tone1 = y_tone1.to(device=device, dtype=torch.float32)
            y_tone2 = y_tone2.to(device=device, dtype=torch.float32)
            subj_id = subj_id.to(device=device, dtype=torch.float32)
            y_idx = y_idx.to(device=device, dtype=torch.int64)  # ← Move y_idx to device

            # Prepare targets dict with ALL FOUR task targets
            targets_dict = {
                'semantic': y_semantic,
                'visual': y_visual,
                'acoustic': [y_tone1, y_tone2],
                'y_true': y_idx,  # ← Add word labels for e2e classification (NEW)
            }

            # Create token mask
            token_len = params.model.encoder.emb_len
            token_mask = torch.ones((X.shape[0], token_len), dtype=torch.float32, device=device)

            # Forward pass
            inputs = [X, targets_dict, subj_id, token_mask]
            outputs, loss = model(inputs)

            batch_size_i = X.shape[0]

            # Aggregate losses (include e2e classification loss, but not cls)
            for key_i in ['total', 'semantic', 'visual', 'acoustic', 'e2e']:  # ← Only record e2e, not cls
                if hasattr(loss, key_i):
                    loss_val = getattr(loss, key_i).item()
                    if hasattr(loss_dict, key_i):
                        loss_dict[key_i].append(np.array([loss_val, batch_size_i], dtype=np.float32))
                    else:
                        loss_dict[key_i] = [np.array([loss_val, batch_size_i], dtype=np.float32)]

            # Collect embeddings
            all_semantic_embs.append(outputs['semantic'].detach().cpu().numpy())
            all_visual_embs.append(outputs['visual'].detach().cpu().numpy())
            all_semantic_labels.append(y_semantic.detach().cpu().numpy())
            all_visual_labels.append(y_visual.detach().cpu().numpy())
            all_y_idx.append(y_idx.detach().cpu().numpy())  # ← Collect integer labels

            # Collect e2e classification predictions (NEW)
            if 'y_pred' in outputs:
                all_y_pred.append(outputs['y_pred'].detach().cpu().numpy())

            # Collect acoustic predictions (average across token dimension)
            tone1_pred = outputs['acoustic'][0].detach().cpu().numpy()  # (batch, token_len, n_tones)
            tone2_pred = outputs['acoustic'][1].detach().cpu().numpy()
            tone1_pred = np.mean(tone1_pred, axis=1)  # (batch, n_tones)
            tone2_pred = np.mean(tone2_pred, axis=1)
            all_tone1_preds.append(tone1_pred)
            all_tone2_preds.append(tone2_pred)
            all_tone1_labels.append(y_tone1.detach().cpu().numpy())
            all_tone2_labels.append(y_tone2.detach().cpu().numpy())

    # Average losses
    for key_i, item_i in loss_dict.items():
        item_i = np.stack(item_i, axis=0)
        item_i = np.sum(item_i[:, 0] * item_i[:, 1]) / np.sum(item_i[:, 1])
        loss_dict[key_i] = item_i

    # Concatenate embeddings
    semantic_embeddings = np.concatenate(all_semantic_embs, axis=0)
    visual_embeddings = np.concatenate(all_visual_embs, axis=0)
    semantic_labels = np.concatenate(all_semantic_labels, axis=0)
    visual_labels = np.concatenate(all_visual_labels, axis=0)
    y_idx_all = np.concatenate(all_y_idx, axis=0)  # ← Concatenate integer labels

    # Calculate acoustic accuracy
    tone1_preds = np.concatenate(all_tone1_preds, axis=0)
    tone2_preds = np.concatenate(all_tone2_preds, axis=0)
    tone1_labels = np.concatenate(all_tone1_labels, axis=0)
    tone2_labels = np.concatenate(all_tone2_labels, axis=0)

    tone1_acc = np.mean(np.argmax(tone1_preds, axis=-1) == np.argmax(tone1_labels, axis=-1))
    tone2_acc = np.mean(np.argmax(tone2_preds, axis=-1) == np.argmax(tone2_labels, axis=-1))

    acoustic_results = {
        'tone1_acc': tone1_acc,
        'tone2_acc': tone2_acc,
        'tone1_preds': tone1_preds,
        'tone2_preds': tone2_preds,
        'tone1_labels': tone1_labels,
        'tone2_labels': tone2_labels,
    }

    # Calculate e2e classification accuracy (NEW)
    e2e_results = {}
    if len(all_y_pred) > 0:
        y_pred_all = np.concatenate(all_y_pred, axis=0)  # (n_samples, n_labels)
        y_pred_class = np.argmax(y_pred_all, axis=-1)  # (n_samples,)
        e2e_acc = np.mean(y_pred_class == y_idx_all)
        e2e_results = {
            'e2e_acc': e2e_acc,
            'y_pred': y_pred_all,
            'y_true': y_idx_all,
        }

    return loss_dict, semantic_embeddings, visual_embeddings, semantic_labels, visual_labels, acoustic_results, e2e_results, y_idx_all  # ← Return e2e_results

# def train func
def train():
    """
    Train `duin_multitask` model.

    Args:
        None

    Returns:
        None
    """
    global model, optimizer, params, paths

    # Load pretrained model params to get n_channels and n_subjects (following run_align_semantic.py pattern)
    path_pt_ckpt = os.path.join(
        params.train.base, params.train.pt_ckpt
    ) if params.train.pt_ckpt is not None else None
    path_pt_params = os.path.join(
        params.train.base, *params.train.pt_ckpt.split(os.sep)[:-2], "save", "params"
    ) if params.train.pt_ckpt is not None else None
    # Load `n_subjects` & `n_channels` & `d_output` from `path_pt_params`.
    if path_pt_params is not None:
        params_pt = load_pickle(path_pt_params)
        n_subjects = params_pt.model.n_subjects
        n_channels = params_pt.model.n_channels
        d_output = params_pt.model.subj.d_output
        # Update params immediately to ensure correct model initialization
        params.model.subj.d_output = d_output
    else:
        params_pt = None
        n_subjects = params.model.n_subjects
        n_channels = params.model.n_channels
        d_output = params.model.subj.d_output

    if params.train.dataset == "seeg_he2023xuanwu":
        # Initialize the configurations of all subjects
        subjs_cfg_all = {
            "001": utils.DotDict({
                "name": "001", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "001"),
                "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
            }),
            "002": utils.DotDict({
                "name": "002", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "002"),
                "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
            }),
            "003": utils.DotDict({
                "name": "003", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "003"),
                "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"],
            }),
            "004": utils.DotDict({
                "name": "004", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "004"),
                "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
            }),
            "005": utils.DotDict({
                "name": "005", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "005"),
                "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
            }),
            "006": utils.DotDict({
                "name": "006", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "006"),
                "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
            }),
            "007": utils.DotDict({
                "name": "007", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "007"),
                "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
            }),
            "008": utils.DotDict({
                "name": "008", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "008"),
                "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
            }),
            "009": utils.DotDict({
                "name": "009", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "009"),
                "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
            }),
            "010": utils.DotDict({
                "name": "010", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "010"),
                "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
            }),
            "011": utils.DotDict({
                "name": "011", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "011"),
                "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
            }),
            "012": utils.DotDict({
                "name": "012", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "012"),
                "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
            }),
        }
        # Filter to only the subjects we're training on
        subjs_cfg = [subjs_cfg_all[subj_i] for subj_i in params.train.subjs]
        subj_idxs = params.train.subj_idxs
        assert len(subj_idxs) == len(subjs_cfg)

        # Create load_params
        load_params = utils.DotDict({
            "name": "joint-multitask-cls-semantic-visual-acoustic-e2e",
            "type": "bipolar_default",  # Use underscore, not dot
            "permutation": False,
            "resample_rate": 1000,
            "task": "word_recitation",
            "use_align": False,  # Use unaligned data (same as other alignment tasks)
            "n_channels": n_channels,
            "n_subjects": n_subjects,
            "subj_idxs": subj_idxs,
            "subjs_cfg": subjs_cfg,
        })
    else:
        raise ValueError(f"ERROR: Unknown dataset {params.train.dataset} in train.duin.run_multitask.")

    # Load data
    dataset_train, dataset_validation, dataset_test = load_data(load_params)

    # Initialize model device - set to GPU if available
    params.model.device = torch.device("cuda:{:d}".format(0)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] Using device: {params.model.device}")
    paths.run.logger.summaries.info(f"Using device: {params.model.device}")

    # Debug: Print params before model creation
    print(f"[DEBUG] Before model creation:")
    print(f"  params.model.n_channels: {params.model.n_channels}")
    print(f"  params.model.subj.d_input: {params.model.subj.d_input}")
    print(f"  params.model.subj.d_output: {params.model.subj.d_output}")
    print(f"  params.model.n_subjects: {params.model.n_subjects}")
    print(f"  params.model.subj.n_subjects: {params.model.subj.n_subjects}")
    print(f"  Expected SubjectLayer W output: d_input * d_output = {params.model.subj.d_input} * {params.model.subj.d_output} = {params.model.subj.d_input * params.model.subj.d_output}")

    # Initialize model
    model = duin_model(params=params.model)
    if path_pt_ckpt is not None:
        model.load_weight(path_ckpt=path_pt_ckpt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device=params.model.device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.train.lr_i)

    # Initialize best result tracking
    best_val_total_loss = float('inf')
    best_val_semantic_loss = float('inf')
    best_val_visual_loss = float('inf')
    best_val_acoustic_acc = 0.0
    best_epoch = -1
    best_ckpt_path = None

    # Get TensorBoard writer
    writer = paths.run.logger.tensorboard

    # Training loop
    for epoch_idx in range(params.train.n_epochs):
        time_start = time.time()
        params.train.epoch = epoch_idx

        # === Training Phase ===
        model.train()
        loss_train = utils.DotDict()

        for batch_idx, batch_data in enumerate(dataset_train):
            X, y_semantic, y_visual, y_tone1, y_tone2, subj_id, y_idx = batch_data  # ← Unpack y_idx
            X = X.to(device=params.model.device, dtype=torch.float32)
            y_semantic = y_semantic.to(device=params.model.device, dtype=torch.float32)
            y_visual = y_visual.to(device=params.model.device, dtype=torch.float32)
            y_tone1 = y_tone1.to(device=params.model.device, dtype=torch.float32)
            y_tone2 = y_tone2.to(device=params.model.device, dtype=torch.float32)
            subj_id = subj_id.to(device=params.model.device, dtype=torch.float32)
            y_idx = y_idx.to(device=params.model.device, dtype=torch.int64)  # ← Move y_idx to device

            # Prepare targets dict with ALL FOUR task targets
            targets_dict = {
                'semantic': y_semantic,
                'visual': y_visual,
                'acoustic': [y_tone1, y_tone2],
                'y_true': y_idx,  # ← Add word labels for e2e classification
            }

            # Create token mask
            token_len = params.model.encoder.emb_len
            token_mask = torch.ones((X.shape[0], token_len), dtype=torch.float32, device=params.model.device)

            # Forward
            inputs = [X, targets_dict, subj_id, token_mask]
            outputs, loss = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss.total.backward()
            optimizer.step()

            # Aggregate losses (only record e2e, not cls)
            batch_size_i = X.shape[0]
            for key_i in ['total', 'semantic', 'visual', 'acoustic', 'e2e']:
                if hasattr(loss, key_i):
                    loss_val = getattr(loss, key_i).item()
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([loss_val, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([loss_val, batch_size_i], dtype=np.float32)]

        # Average training losses
        for key_i, item_i in loss_train.items():
            item_i = np.stack(item_i, axis=0)
            item_i = np.sum(item_i[:, 0] * item_i[:, 1]) / np.sum(item_i[:, 1])
            loss_train[key_i] = item_i

        # === Validation Phase ===
        loss_validation, val_semantic_embs, val_visual_embs, val_semantic_labels, val_visual_labels, val_acoustic_results, val_e2e_results, val_y_idx = _evaluate(
            model, dataset_validation, params, params.model.device
        )

        # === Test Phase ===
        loss_test, test_semantic_embs, test_visual_embs, test_semantic_labels, test_visual_labels, test_acoustic_results, test_e2e_results, test_y_idx = _evaluate(
            model, dataset_test, params, params.model.device
        )

        # === Log epoch time ===
        time_stop = time.time()
        msg = f"Finish train epoch {epoch_idx} in {time_stop - time_start:.2f} seconds."
        print(msg); paths.run.logger.summaries.info(msg)

        # === Log losses to summaries.log ===
        msg = log_loss_block("train", loss_train)
        print(msg); paths.run.logger.summaries.info(msg)
        msg = log_loss_block("validation", loss_validation)
        print(msg); paths.run.logger.summaries.info(msg)
        msg = log_loss_block("test", loss_test)
        print(msg); paths.run.logger.summaries.info(msg)

        # === Log acoustic accuracy ===
        msg = f"Accuracy(validation): tone1={val_acoustic_results['tone1_acc']:.4f}, tone2={val_acoustic_results['tone2_acc']:.4f}"
        print(msg); paths.run.logger.summaries.info(msg)

        # === Log e2e classification accuracy (NEW) ===
        if 'e2e_acc' in val_e2e_results:
            msg = f"Accuracy(validation): e2e_cls={val_e2e_results['e2e_acc']:.4f}"
            print(msg); paths.run.logger.summaries.info(msg)
            msg = f"Accuracy(test): e2e_cls={test_e2e_results['e2e_acc']:.4f}"
            print(msg); paths.run.logger.summaries.info(msg)
        msg = f"Accuracy(test): tone1={test_acoustic_results['tone1_acc']:.4f}, tone2={test_acoustic_results['tone2_acc']:.4f}"
        print(msg); paths.run.logger.summaries.info(msg)

        # === Log to TensorBoard ===
        # Log losses
        for phase_name, loss_dict in [("train", loss_train), ("validation", loss_validation), ("test", loss_test)]:
            for key_i, loss_i in loss_dict.items():
                writer.add_scalar(os.path.join("losses", phase_name, key_i), loss_i, global_step=epoch_idx)

        # Log acoustic accuracy
        writer.add_scalar("accuracy/validation/tone1", val_acoustic_results['tone1_acc'], global_step=epoch_idx)
        writer.add_scalar("accuracy/validation/tone2", val_acoustic_results['tone2_acc'], global_step=epoch_idx)
        writer.add_scalar("accuracy/test/tone1", test_acoustic_results['tone1_acc'], global_step=epoch_idx)
        writer.add_scalar("accuracy/test/tone2", test_acoustic_results['tone2_acc'], global_step=epoch_idx)

        # Log e2e classification accuracy (NEW)
        if 'e2e_acc' in val_e2e_results:
            writer.add_scalar("accuracy/validation/e2e_cls", val_e2e_results['e2e_acc'], global_step=epoch_idx)
        if 'e2e_acc' in test_e2e_results:
            writer.add_scalar("accuracy/test/e2e_cls", test_e2e_results['e2e_acc'], global_step=epoch_idx)

        # Log learning rate
        writer.add_scalar("learning_rate", params.train.lr_i, global_step=epoch_idx)

        # === Log task weights to TensorBoard ===
        model_to_log = model.module if hasattr(model, 'module') else model
        if params.model.use_uncertainty_weighting:
            # Log learnable uncertainty parameters and derived weights
            log_var_semantic = model_to_log.log_var_semantic.item()
            log_var_visual = model_to_log.log_var_visual.item()
            log_var_acoustic = model_to_log.log_var_acoustic.item()

            # Calculate actual task weights (precision = exp(-log_var))
            weight_semantic = np.exp(-log_var_semantic)
            weight_visual = np.exp(-log_var_visual)
            weight_acoustic = np.exp(-log_var_acoustic)

            # Log log-variance (uncertainty)
            writer.add_scalar("task_weights/log_var_semantic", log_var_semantic, global_step=epoch_idx)
            writer.add_scalar("task_weights/log_var_visual", log_var_visual, global_step=epoch_idx)
            writer.add_scalar("task_weights/log_var_acoustic", log_var_acoustic, global_step=epoch_idx)

            # Log derived task weights (precision)
            writer.add_scalar("task_weights/weight_semantic", weight_semantic, global_step=epoch_idx)
            writer.add_scalar("task_weights/weight_visual", weight_visual, global_step=epoch_idx)
            writer.add_scalar("task_weights/weight_acoustic", weight_acoustic, global_step=epoch_idx)

            # Log normalized weights (for easier interpretation)
            total_weight = weight_semantic + weight_visual + weight_acoustic
            writer.add_scalar("task_weights/normalized_semantic", weight_semantic / total_weight, global_step=epoch_idx)
            writer.add_scalar("task_weights/normalized_visual", weight_visual / total_weight, global_step=epoch_idx)
            writer.add_scalar("task_weights/normalized_acoustic", weight_acoustic / total_weight, global_step=epoch_idx)
        else:
            # Log fixed task weights from params
            writer.add_scalar("task_weights/weight_semantic", params.model.task_weight_semantic, global_step=epoch_idx)
            writer.add_scalar("task_weights/weight_visual", params.model.task_weight_visual, global_step=epoch_idx)
            writer.add_scalar("task_weights/weight_acoustic", params.model.task_weight_acoustic, global_step=epoch_idx)

            # Log normalized weights
            total_weight = params.model.task_weight_semantic + params.model.task_weight_visual + params.model.task_weight_acoustic
            writer.add_scalar("task_weights/normalized_semantic", params.model.task_weight_semantic / total_weight, global_step=epoch_idx)
            writer.add_scalar("task_weights/normalized_visual", params.model.task_weight_visual / total_weight, global_step=epoch_idx)
            writer.add_scalar("task_weights/normalized_acoustic", params.model.task_weight_acoustic / total_weight, global_step=epoch_idx)

        # === Track and save best model ===
        current_val_total_loss = loss_validation.get("total", float('inf'))
        if current_val_total_loss < best_val_total_loss:
            # Remove previous best checkpoint
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
                msg = f"Removed previous best checkpoint: {best_ckpt_path}"
                print(msg); paths.run.logger.summaries.info(msg)

            best_val_total_loss = current_val_total_loss
            best_val_semantic_loss = loss_validation.get("semantic", float('inf'))
            best_val_visual_loss = loss_validation.get("visual", float('inf'))
            best_val_acoustic_acc = (val_acoustic_results['tone1_acc'] + val_acoustic_results['tone2_acc']) / 2
            best_epoch = epoch_idx

            # Save best model checkpoint to summaries directory
            best_ckpt_path = os.path.join(paths.run.summaries, f"best_epoch_{epoch_idx:03d}_loss_{best_val_total_loss:.5f}.pth")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), best_ckpt_path)
            msg = f"New best model saved (epoch {epoch_idx}) with val_loss={best_val_total_loss:.6f}"
            print(msg); paths.run.logger.summaries.info(msg)

            # Save best results to summaries directory (with integer labels as final column)
            best_semantic_with_labels = np.concatenate([test_semantic_embs, test_y_idx[:, None]], axis=1)
            np.save(
                os.path.join(paths.run.summaries, "best_semantic_embeddings.npy"),
                best_semantic_with_labels
            )
            best_visual_with_labels = np.concatenate([test_visual_embs, test_y_idx[:, None]], axis=1)
            np.save(
                os.path.join(paths.run.summaries, "best_visual_embeddings.npy"),
                best_visual_with_labels
            )
            np.savez(
                os.path.join(paths.run.summaries, "best_acoustic_results.npz"),
                tone1_acc=test_acoustic_results['tone1_acc'],
                tone2_acc=test_acoustic_results['tone2_acc'],
                tone1_preds=test_acoustic_results['tone1_preds'],
                tone2_preds=test_acoustic_results['tone2_preds'],
                tone1_labels=test_acoustic_results['tone1_labels'],
                tone2_labels=test_acoustic_results['tone2_labels']
            )
            msg = f"Best results saved to {paths.run.summaries}"
            print(msg); paths.run.logger.summaries.info(msg)

        # === Save checkpoint and results every 50 epochs ===
        if (epoch_idx + 1) % 50 == 0 or (epoch_idx + 1) == params.train.n_epochs:
            # Save checkpoint
            ckpt_path = os.path.join(paths.run.ckpt, f"checkpoint-{epoch_idx:03d}.pth")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), ckpt_path)
            msg = f"INFO: Saved checkpoint to {ckpt_path}"
            print(msg); paths.run.logger.summaries.info(msg)

            # Save semantic embeddings with integer labels as final column (compatible with evaluation script)
            semantic_emb_with_labels = np.concatenate([test_semantic_embs, test_y_idx[:, None]], axis=1)  # (n_samples, 769)
            semantic_emb_path = os.path.join(paths.run.save_embeddings, f"semantic_embeddings_epoch_{epoch_idx + 1:03d}.npy")
            np.save(semantic_emb_path, semantic_emb_with_labels)
            msg = f"INFO: Saved semantic embeddings to {semantic_emb_path} (shape: {semantic_emb_with_labels.shape})"
            print(msg); paths.run.logger.summaries.info(msg)

            # Save visual embeddings with integer labels as final column (compatible with evaluation script)
            visual_emb_with_labels = np.concatenate([test_visual_embs, test_y_idx[:, None]], axis=1)  # (n_samples, 769)
            visual_emb_path = os.path.join(paths.run.save_embeddings, f"visual_embeddings_epoch_{epoch_idx + 1:03d}.npy")
            np.save(visual_emb_path, visual_emb_with_labels)
            msg = f"INFO: Saved visual embeddings to {visual_emb_path} (shape: {visual_emb_with_labels.shape})"
            print(msg); paths.run.logger.summaries.info(msg)

            # Save acoustic results
            acoustic_result_path = os.path.join(paths.run.save_embeddings, f"acoustic_results_epoch_{epoch_idx + 1:03d}.npz")
            np.savez(
                acoustic_result_path,
                tone1_acc=test_acoustic_results['tone1_acc'],
                tone2_acc=test_acoustic_results['tone2_acc'],
                tone1_preds=test_acoustic_results['tone1_preds'],
                tone2_preds=test_acoustic_results['tone2_preds'],
                tone1_labels=test_acoustic_results['tone1_labels'],
                tone2_labels=test_acoustic_results['tone2_labels']
            )
            msg = f"INFO: Saved acoustic results to {acoustic_result_path}"
            print(msg); paths.run.logger.summaries.info(msg)

        # Update learning rate
        params.iteration(epoch_idx + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = params.train.lr_i

    # === Final summary ===
    msg = f"\n{'=' * 50}\nTraining Complete!\n{'=' * 50}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best epoch: {best_epoch}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best validation total loss: {best_val_total_loss:.6f}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best validation semantic loss: {best_val_semantic_loss:.6f}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best validation visual loss: {best_val_visual_loss:.6f}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best validation acoustic accuracy: {best_val_acoustic_acc:.4f}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best model saved at: {best_ckpt_path}"
    print(msg); paths.run.logger.summaries.info(msg)
    msg = f"Best results saved in: {paths.run.summaries}"
    print(msg); paths.run.logger.summaries.info(msg)

    # Close TensorBoard writer
    writer.close()

"""
arg funcs
"""
# def get_args_parser func
def get_args_parser():
    """
    Parse arguments from command line.

    Args:
        None

    Returns:
        parser: ArgumentParser - The parser of arguments.
    """
    # Initialize parser.
    parser = argparse.ArgumentParser("Joint Multi-task + E2E Classification Training for Du-IN", add_help=False)
    # Add arguments.
    parser.add_argument("--base", type=str, default=os.getcwd())
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--subjs", type=str, default="001")
    parser.add_argument("--subj_idxs", type=str, default="0")
    parser.add_argument("--pt_ckpt", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    # Multi-task parameters (including e2e classification)
    parser.add_argument("--task_weight_semantic", type=float, default=1.0)
    parser.add_argument("--task_weight_visual", type=float, default=1.0)
    parser.add_argument("--task_weight_acoustic", type=float, default=1.0)
    parser.add_argument("--task_weight_e2e", type=float, default=2.0)  # NEW: e2e task weight
    parser.add_argument("--use_uncertainty_weighting", action="store_true")
    parser.add_argument("--acoustic_use_contra", action="store_true")
    # Loss scales
    parser.add_argument("--semantic_align_loss_scale", type=float, default=5.0)
    parser.add_argument("--semantic_contra_loss_scale", type=float, default=0.5)
    parser.add_argument("--visual_align_loss_scale", type=float, default=5.0)
    parser.add_argument("--visual_contra_loss_scale", type=float, default=0.5)
    parser.add_argument("--acoustic_cls_loss_scale", type=float, default=1.0)
    parser.add_argument("--acoustic_contra_loss_scale", type=float, default=0.5)
    parser.add_argument("--cls_loss_scale", type=float, default=1.0)  # NEW: e2e classification loss scale
    # Head architectures
    parser.add_argument("--semantic_d_hidden", type=str, default="2048,1024,768")
    parser.add_argument("--semantic_dropout", type=float, default=0.1)
    parser.add_argument("--visual_d_hidden", type=str, default="2048,1024,768")
    parser.add_argument("--visual_dropout", type=float, default=0.1)
    parser.add_argument("--acoustic_d_hidden", type=str, default="128")
    parser.add_argument("--acoustic_dropout", type=float, default=0.5)
    # Fusion head parameters (NEW)
    parser.add_argument("--fusion_d_hidden", type=str, default="512,256")
    parser.add_argument("--fusion_dropout", type=float, default=0.3)
    # Encoder dropout
    parser.add_argument("--attn_dropout", type=float, default=0.25)
    parser.add_argument("--ff_dropout", type=str, default="0.25,0.1")
    # Contrastive parameters
    parser.add_argument("--contra_d_hidden", type=int, default=32)
    parser.add_argument("--contra_loss_mode", type=str, default="clip_orig")
    # Encoder architecture
    parser.add_argument("--n_blocks", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    # Run script path
    parser.add_argument("--run_script", type=str, default=None)
    # Return parser.
    return parser

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    # Parse arguments.
    parser = get_args_parser(); args = parser.parse_args()
    # Process arguments to get list of configurations.
    seeds = [int(seed_i) for seed_i in args.seeds.split(",")]
    subjs = [subj_i for subj_i in args.subjs.split(",")]
    subj_idxs = [int(subj_idx_i) for subj_idx_i in args.subj_idxs.split(",")]

    # Determine the project root directory (go up two levels from train/duin)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # Get params from specified configuration.
    from params.duin_params import duin_joint_multitask_cls_params
    # Execute experiments.
    for seed_i in seeds:
        for subj_i, subj_idx_i in zip(subjs, subj_idxs):
            # Initialize random seed.
            utils.model.torch.set_seeds(seed_i)
            # Initialize params.
            params_i = duin_joint_multitask_cls_params(dataset="seeg_he2023xuanwu")
            # Use project root if args.base is the default, otherwise use args.base
            params_i.train.base = project_root if args.base == os.getcwd() else args.base
            params_i.train.seed = seed_i
            params_i.train.subjs = [subj_i]
            params_i.train.subj_idxs = [subj_idx_i]
            params_i.train.pt_ckpt = args.pt_ckpt
            params_i.train.n_epochs = args.n_epochs
            params_i.train.warmup_epochs = args.warmup_epochs
            params_i.train.batch_size = args.batch_size
            params_i.train.lr_factors = (args.lr_min, args.lr_max)
            params_i.train.epoch_start = 0
            params_i.train.run_script = args.run_script
            # Multi-task parameters (including e2e)
            params_i.model.task_weight_semantic = args.task_weight_semantic
            params_i.model.task_weight_visual = args.task_weight_visual
            params_i.model.task_weight_acoustic = args.task_weight_acoustic
            params_i.model.task_weight_e2e = args.task_weight_e2e  # NEW
            params_i.model.use_uncertainty_weighting = args.use_uncertainty_weighting
            params_i.model.acoustic_use_contra = args.acoustic_use_contra
            # Loss scales
            params_i.model.semantic_align_loss_scale = args.semantic_align_loss_scale
            params_i.model.semantic_contra_loss_scale = args.semantic_contra_loss_scale
            params_i.model.visual_align_loss_scale = args.visual_align_loss_scale
            params_i.model.visual_contra_loss_scale = args.visual_contra_loss_scale
            params_i.model.acoustic_cls_loss_scale = args.acoustic_cls_loss_scale
            params_i.model.acoustic_contra_loss_scale = args.acoustic_contra_loss_scale
            params_i.model.cls_loss_scale = args.cls_loss_scale  # NEW
            # Head architectures
            params_i.model.semantic_align.d_hidden = [int(d) for d in args.semantic_d_hidden.split(",")]
            params_i.model.semantic_align.dropout = args.semantic_dropout
            params_i.model.visual_align.d_hidden = [int(d) for d in args.visual_d_hidden.split(",")]
            params_i.model.visual_align.dropout = args.visual_dropout
            params_i.model.acoustic_cls.d_hidden = [int(d) for d in args.acoustic_d_hidden.split(",")]
            params_i.model.acoustic_cls.dropout = args.acoustic_dropout
            # Fusion head parameters (NEW)
            params_i.model.fusion.d_hidden = [int(d) for d in args.fusion_d_hidden.split(",")]
            params_i.model.fusion.dropout = args.fusion_dropout
            # Encoder dropout
            params_i.model.encoder.attn_dropout = args.attn_dropout
            ff_dropout = [float(d) for d in args.ff_dropout.split(",")]
            params_i.model.encoder.ff_dropout = ff_dropout
            # Contrastive parameters
            params_i.model.contra.d_contra = args.contra_d_hidden
            params_i.model.contra.loss_mode = args.contra_loss_mode
            # Encoder architecture
            params_i.model.encoder.n_blocks = args.n_blocks
            params_i.model.encoder.n_heads = args.n_heads
            # Initialize training process.
            init(params_=params_i)
            # Execute training process.
            train()
