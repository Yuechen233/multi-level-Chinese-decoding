#!/usr/bin/env python3
"""
Created on 21:51, Jan. 19th, 2024

@author: Norbert Zheng
"""
import re, torch
import copy as cp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "duin_vqvae",
    "duin_mae",
    "duin_cls",
    "duin_align",
    "duin_llm",
    "duin_acoustic_cls",
    "duin_multitask",
    "duin_fusion_cls",
    "duin_joint_multitask_cls",
]

# def duin_vqvae class
class duin_vqvae(nn.Module):
    """
    DuIN model for neural signal prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_vqvae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_vqvae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_vqvae, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None) and (self.params.decoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize decoder block.
        # decoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.decoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.decoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize regression block.
        # rgs_block - (batch_size, token_len, d_model) -> (batch_size, seq_len, d_neural)
        self.rgs_block = TimeRGSHead(params=self.params.rgs)
        # Initialize de-subject block.
        # desubj_block - (batch_size, seq_len, d_neural) -> (batch_size, seq_len, n_channels)
        self.desubj_block = SubjectBlock(params=self.params.desubj)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*vq_block": "vq_block",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_vqvae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_vqvae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,subj_id,channel_mask].

        Returns:
            X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
        X = inputs[0]; subj_id = inputs[1]; channel_mask = inputs[2]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward decoder & regression block to get the corresponding reconstructon.
        # TODO: Support subject-layer in `rgs_block`, we do not reconstruct the intermediate `X_h_reconstr`.
        # T_reconstr - (batch_size, token_len, d_model)
        T_reconstr = torch.reshape(self.decoder(E_vq), shape=token_shape)
        # X_reconstr - (batch_size, seq_len, n_channels)
        X_reconstr = self.desubj_block((self.rgs_block(T_reconstr), subj_id))
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(X_reconstr, X, weight=channel_mask.to(dtype=X.dtype))
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.rgs_loss_scale * loss_rgs +\
            self.params.vq_loss_scale * loss_vq
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "vq": loss_vq,
            "rgs": loss_rgs,
        })
        # Return the final `X_reconstr` & `loss`.
        return X_reconstr, loss

    # def quantize func
    def quantize(self, inputs):
        """
        Forward `duin_vqvae` to get the quantized embeddings.

        Args:
            inputs: tuple - The input data, including [X,subj_id].

        Returns:
            E_vq: (batch_size, emb_len, d_model) - The quantized embeddings.
            loss_vq: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, emb_len, codex_size) - The one-hot probabilities of the embeddings.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; subj_id = inputs[1]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
        E_vq, loss_vq, codex_probs = self.vq_block(E)
        # Return the final `E_vq` & `loss_vq` & `codex_probs`.
        return E_vq, loss_vq, codex_probs

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target, weight=None):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, seq_len, n_channels) - Value of the object.
            target: (batch_size, seq_len, n_channels) - Traget of the object.
            weight: (batch_size, n_channels) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, seq_len, n_channels)
        loss = torch.square(target - value)
        # Average over all locations.
        # loss - (batch_size, n_channels)
        loss = torch.mean(torch.flatten(torch.permute(loss, dims=[0,-1,*range(1, len(loss.shape)-1)]), start_dim=2, end_dim=-1), dim=-1)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

# def duin_mae class
class duin_mae(nn.Module):
    """
    DuIN model for neural token prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_mae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_mae, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize mask embedding.
        # mask_emb - (d_model,)
        mask_emb = torch.ones((self.params.encoder.d_model,), dtype=torch.float32)
        self.mask_emb = nn.Parameter(mask_emb, requires_grad=True)
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_tokens)
        self.cls_block = TokenCLSHead(params=self.params.cls)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `mask_emb`.
        nn.init.trunc_normal_(self.mask_emb, mean=0., std=0.02)

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,c_true,subj_id].

        Returns:
            c_pred: (batch_size, emb_len, codex_size) - The predicted codex.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
        X = inputs[0]; c_true = inputs[1]; subj_id = inputs[2]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E_init - (batch_size, emb_len, d_model)
        E_init = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Generate mask according to the init embedding `E`.
        # mask - (batch_size, emb_len)
        mask = self.gen_mask(E_init, mask_ratio=self.params.mask_ratio)
        # Get the masked embedding `E_masked` according to `mask`.
        # mask_emb - (batch_size, emb_len, d_model)
        mask_emb = self.mask_emb[None,None,...].expand(*mask.shape, -1)
        # E_masked - (2[list], batch_size, emb_len, d_model)
        E_masked = [
            (E_init * (1. - mask[...,None].to(dtype=E_init.dtype)) + mask_emb * mask[...,None].to(dtype=E_init.dtype)),
            (E_init * mask[...,None].to(dtype=E_init.dtype) + mask_emb * (1. - mask[...,None].to(dtype=E_init.dtype))),
        ]
        # Forward encoder block to get time-aligned token sequence.
        # E - (2[list], batch_size, emb_len, d_model)
        E = [self.encoder(E_i) for E_i in E_masked]
        # Forward classification block to get the corresponding prediction.
        # c_pred - (batch_size, emb_len, codex_size)
        c_pred = [self.cls_block(E_i) for E_i in E]
        c_pred = (
            (c_pred[0] * mask[...,None].to(dtype=c_pred[0].dtype)) +\
            (c_pred[1] * (1. - mask[...,None].to(dtype=c_pred[1].dtype)))
        )
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(
            torch.reshape(c_pred, shape=(-1, c_pred.shape[-1])),
            torch.reshape(c_true, shape=(-1, c_true.shape[-1])),
        )
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `c_pred` & `loss`.
        return c_pred, loss

    # def gen_mask func
    def gen_mask(self, E, mask_ratio=0.5):
        """
        Generate mask for embedding sequence.

        Args:
            E: (batch_size, emb_len, d_model) - The embedding sequence.
            mask_ratio: float - The mask ratio of each embedding item.

        Returns:
            mask: (batch_size, emb_len) - The generated mask.
        """
        # Initialize `batch_size` & `emb_len` & `d_model` from `E`.
        batch_size, emb_len, d_model = E.shape
        # Initialize the length of keep embedding items.
        keep_len = int(emb_len * (1. - mask_ratio))
        # Initialize the noise for further argsort.
        # noise - (batch_size, emb_len)
        noise = torch.rand((batch_size, emb_len), dtype=E.dtype).to(device=E.device)
        # Get the corresponding `shuffle_idxs` & `restore_idxs`.
        # Note: `torch.argsort` is reversible, we have `shuffle_idxs = torch.argsort(restore_idxs)`.
        shuffle_idxs = torch.argsort(noise, dim=-1); restore_idxs = torch.argsort(shuffle_idxs, dim=-1)
        # Generate the bool mask: `False` is keep, `True` is remove.
        # mask - (batch_size, emb_len)
        mask = torch.ones((batch_size, emb_len), dtype=torch.bool).to(device=E.device); mask[:,:keep_len] = False
        # Un-shuffle to get the bool mask.
        mask = torch.gather(mask, dim=-1, index=restore_idxs)
        # Return the final `mask`.
        return mask

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

# def duin_cls class
class duin_cls(nn.Module):
    """
    DuIN model for classification task.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_cls, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize contrastive block.
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, n_labels)
        self.cls_block = LabelCLSHead(params=self.params.cls)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        #ckpt_dict = torch.load(path_ckpt)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(path_ckpt, map_location=device)

        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true,subj_id].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; y_true = inputs[1]; subj_id = inputs[2]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Calculate the contrastive loss.
        # loss_contra - torch.float32
        loss_contra, _ = self.contra_block(((E, E), (y_true, y_true)))
        # Forward classification block to get the corresponding prediction.
        # y_pred - (batch_size, n_labels)
        y_pred = self.cls_block(E)
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(y_pred, y_true)
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls +\
            self.params.contra_loss_scale * loss_contra
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
            "contra": loss_contra,
        })
        # Return the final `y_pred` & `loss`.
        return y_pred, loss

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

# def duin_align class
class duin_align(nn.Module):   # 新增的alignment任务模型
    """
    DuIN model for alignment task.
    Projects brain embeddings to external embedding spaces (e.g., CLIP embeddings).
    """

    def __init__(self, params, **kwargs):   # 不变
        """
        Initialize `duin_align` object.

        Args:
            params: DotDict - Model parameters initialized by duin_align_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_align, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):   # 删除分类头，新增投影头
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize contrastive block.
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, n_labels)
        #self.cls_block = LabelCLSHead(params=self.params.cls)
        self.align_head = AlignHead(params=self.params.align)   # 新增

    # def _init_weight func
    def _init_weight(self):   # 不变
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):   # 不变
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        #ckpt_dict = torch.load(path_ckpt)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(path_ckpt, map_location=device)

        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_align."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):   # 修改最后一层的输出

        X = inputs[0]; Y_target = inputs[1]; subj_id = inputs[2]  

        X_h = self.subj_block((X, subj_id))
        T = self.tokenizer(X_h); token_shape = T.shape
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        E = self.encoder(E)
        E_vq, loss_vq, _ = self.vq_block(E)

        Z_pred = self.align_head(E)   # 新投影头输出

        # === ✅ 在计算任何loss前，进行L2归一化 ===
        Z_pred_norm = torch.nn.functional.normalize(Z_pred, p=2, dim=-1)
        Y_target_norm = torch.nn.functional.normalize(Y_target, p=2, dim=-1)

        E_norm = torch.nn.functional.normalize(E, p=2, dim=-1)
        loss_contra, _ = self.contra_block(((E_norm, E_norm), (Y_target_norm, Y_target_norm)))

        loss_align = self._loss_align(Z_pred_norm, Y_target_norm)   # 后面定义新损失函数_loss_align(self, value, target)

        loss_total = (
            self.params.align_loss_scale * loss_align + self.params.contra_loss_scale * loss_contra   
        )

        loss = DotDict({
            "total": loss_total,
            "align": loss_align,
            "contra": loss_contra,
        })

        #return y_pred, loss
        return Z_pred_norm, loss  

    """
    loss funcs
    """
    # def _loss_align func
    def _loss_align(self, value, target):   # 新损失函数：MSE(归一化)
        """
        Calculates alignment loss (MSE) between predicted and target embeddings.

        Args:
            value:  (batch_size, d_output) - Predicted embedding from the model.
            target: (batch_size, d_output) - Ground-truth or teacher embedding.

        Returns:
            loss: torch.float32 - Mean squared error between L2-normalized embeddings.
        """
        # ---- Step 1. Validate shapes ----
        assert value.shape == target.shape, f"Shape mismatch: {value.shape} vs {target.shape}"

        # ---- Step 2. Compute MSE loss ----
        # MSE on normalized embeddings is equivalent to minimizing cosine distance
        # value_np = value.detach().cpu().numpy()
        # target_np = target.detach().cpu().numpy()
        # res = [value_origin, value_np, target_np]
        # np.save('emb_res_1.npy', res)
        # print("Saved normalized embeddings to 'emb_res.npy'.")
        # exit()
        loss = 1000 * F.mse_loss(value, target, reduction="mean")

        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()


# def duin_llm class
class duin_llm(nn.Module):
    """
    DuIN model for open-set language decoding task.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_llm` object.

        Args:
            params: DotDict - Model parameters initialized by duin_llm_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_llm, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize classification blocks.
        # cls_blocks - (batch_size, token_len, d_model) -> (batch_size, token_len, n_phonemes)
        cls_initials_params = cp.deepcopy(self.params.cls); cls_initials_params.n_tokens = cls_initials_params.n_initials
        cls_finals_params = cp.deepcopy(self.params.cls); cls_finals_params.n_tokens = cls_finals_params.n_finals
        self.cls_blocks = nn.ModuleList(modules=[
            TokenCLSHead(params=cls_initials_params),
            TokenCLSHead(params=cls_finals_params),
        ])

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_llm."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_llm` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,p_true,subj_id,token_mask].

        Returns:
            p_pred: (2[list], batch_size, token_len, n_phonemes) - The predicted phonemes.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); p_true - (2[list], batch_size, token_len, n_phonemes)
        # subj_id - (batch_size, n_subjects); token_mask - (batch_size, token_len)
        X = inputs[0]; p_true = inputs[1]; subj_id = inputs[2]; token_mask = inputs[3]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward classification block to get the prediction phonemes.
        # p_pred - (2[list], batch_size, token_len, n_phonemes)
        p_pred = [self.cls_blocks[phoneme_idx](E) for phoneme_idx in range(len(self.cls_blocks))]
        # Calculate the classification loss.
        # loss_cls - torch.float32
        weight = token_mask.to(dtype=p_pred[0].dtype)
        #loss_cls = torch.mean(torch.stack([self._loss_cls(p_pred_i, p_true_i, weight=weight)\
        #    for p_pred_i, p_true_i in zip(p_pred, p_true)], dim=0))
        loss_cls = [self._loss_cls(p_pred_i, p_true_i, weight=weight)\
            for p_pred_i, p_true_i in zip(p_pred, p_true)][1]
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `p_pred` & `loss`.
        return p_pred, loss

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target, weight=None):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, emb_len, d_llm) - Value of the object.
            target: (batch_size, emb_len, d_llm) - Traget of the object.
            weight: (batch_size, emb_len, d_llm) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, emb_len, d_llm)
        loss = torch.square(target - value)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

    # def _loss_cls func
    def _loss_cls(self, value, target, weight=None):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, emb_len, n_words) - Value of the object.
            target: (batch_size, emb_len, n_words) - Target of the object.
            weight: (batch_size, d_llm) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Initialize `batch_size` & `emb_len` & `n_words` from `value`.
        batch_size, emb_len, n_words = value.shape
        # Calculate the cross-entropy loss.
        # loss - (batch_size, emb_len)
        loss = torch.reshape(F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=torch.reshape(value, shape=(-1, n_words)), target=torch.reshape(target, shape=(-1, n_words)),
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="none", label_smoothing=0.
        ), shape=(batch_size, emb_len))
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

class duin_acoustic_cls(nn.Module):
    """
    DuIN model for acoustic tone classification task.
    Predicts two tone labels (5 classes each) for Chinese word reading.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_acoustic_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_acoustic_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_acoustic_cls, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize classification blocks for tone1 and tone2.
        # cls_blocks - (batch_size, token_len, d_model) -> (batch_size, token_len, n_tones)
        cls_tone1_params = cp.deepcopy(self.params.cls); cls_tone1_params.n_tokens = cls_tone1_params.n_tone1
        cls_tone2_params = cp.deepcopy(self.params.cls); cls_tone2_params.n_tokens = cls_tone2_params.n_tone2
        self.cls_blocks = nn.ModuleList(modules=[
            TokenCLSHead(params=cls_tone1_params),
            TokenCLSHead(params=cls_tone2_params),
        ])

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_acoustic_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_acoustic_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,t_true,subj_id,token_mask].

        Returns:
            t_pred: (2[list], batch_size, token_len, n_tones) - The predicted tones.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); t_true - (2[list], batch_size, token_len, n_tones)
        # subj_id - (batch_size, n_subjects); token_mask - (batch_size, token_len)
        X = inputs[0]; t_true = inputs[1]; subj_id = inputs[2]; token_mask = inputs[3]
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward classification block to get the prediction tones.
        # t_pred - (2[list], batch_size, token_len, n_tones)
        t_pred = [self.cls_blocks[tone_idx](E) for tone_idx in range(len(self.cls_blocks))]
        # Expand t_true from (batch_size, n_tones) to (batch_size, token_len, n_tones)
        # by repeating across the token dimension for per-token supervision
        token_len = t_pred[0].shape[1]
        t_true_expanded = [t_true_i.unsqueeze(1).expand(-1, token_len, -1) for t_true_i in t_true]
        # Calculate the classification loss.
        # loss_cls - torch.float32
        weight = token_mask.to(dtype=t_pred[0].dtype)
        loss_cls_list = [self._loss_cls(t_pred_i, t_true_expanded_i, weight=weight)\
            for t_pred_i, t_true_expanded_i in zip(t_pred, t_true_expanded)]
        loss_cls = torch.mean(torch.stack(loss_cls_list, dim=0))
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
            "cls_tone1": loss_cls_list[0],
            "cls_tone2": loss_cls_list[1],
        })
        # Return the final `t_pred` & `loss`.
        return t_pred, loss

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target, weight=None):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        NOTE: For acoustic tone classification, L2 normalization can be applied to the logits
        before computing cross-entropy loss (controlled by params.cls.use_l2_norm).
        This helps prevent overfitting by constraining the magnitude of predictions
        and encouraging more confident, stable predictions.

        Args:
            value: (batch_size, emb_len, n_tones) - Value of the object.
            target: (batch_size, emb_len, n_tones) - Target of the object.
            weight: (batch_size, d_llm) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Initialize `batch_size` & `emb_len` & `n_tones` from `value`.
        batch_size, emb_len, n_tones = value.shape

        # Apply L2 normalization to logits if enabled (for overfitting prevention)
        # This helps prevent overfitting by constraining prediction magnitudes
        # Normalize along the last dimension (n_tones) with eps for numerical stability
        if self.params.cls.use_l2_norm:
            value_normalized = F.normalize(value, p=2, dim=-1, eps=1e-12)
        else:
            value_normalized = value

        # Calculate the cross-entropy loss with (optionally normalized) logits.
        # loss - (batch_size, emb_len)
        loss = torch.reshape(F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=torch.reshape(value_normalized, shape=(-1, n_tones)), target=torch.reshape(target, shape=(-1, n_tones)),
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="none", label_smoothing=0.
        ), shape=(batch_size, emb_len))
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

# def duin_multitask class
class duin_multitask(nn.Module):
    """
    DuIN multi-task learning model combining semantic, visual, and acoustic alignment tasks.
    Shares encoder across all three tasks with task-specific heads.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_multitask` object.

        Args:
            params: DotDict - Model parameters initialized by duin_multitask_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        super(duin_multitask, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize shared encoder components
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )

        # Initialize VQ block (used for contrastive learning in alignment tasks)
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )

        # Initialize contrastive block (shared by alignment tasks)
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)

        # Initialize task-specific heads
        # 1. Semantic alignment head - (batch_size, emb_len, d_model) -> (batch_size, 768)
        self.semantic_head = AlignHead(params=self.params.semantic_align)

        # 2. Visual alignment head - (batch_size, emb_len, d_model) -> (batch_size, 768)
        self.visual_head = AlignHead(params=self.params.visual_align)

        # 3. Acoustic classification heads - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_tones)
        cls_tone1_params = cp.deepcopy(self.params.acoustic_cls); cls_tone1_params.n_tokens = cls_tone1_params.n_tone1
        cls_tone2_params = cp.deepcopy(self.params.acoustic_cls); cls_tone2_params.n_tokens = cls_tone2_params.n_tone2
        self.acoustic_heads = nn.ModuleList(modules=[
            TokenCLSHead(params=cls_tone1_params),
            TokenCLSHead(params=cls_tone2_params),
        ])

        # Initialize learnable task weights for uncertainty-based multi-task learning (optional)
        if self.params.use_uncertainty_weighting:
            # Log-variance parameters for automatic task balancing (Kendall et al., 2018)
            self.log_var_semantic = nn.Parameter(torch.zeros(1))
            self.log_var_visual = nn.Parameter(torch.zeros(1))
            self.log_var_acoustic = nn.Parameter(torch.zeros(1))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the specified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(path_ckpt, map_location=device)

        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_multitask."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_multitask` to get predictions and losses for all three tasks.

        Args:
            inputs: tuple - The input data [X, targets_dict, subj_id, token_mask]
                X: (batch_size, seq_len, n_channels) - Brain signals
                targets_dict: dict - Ground-truth targets for each task
                    'semantic': (batch_size, 768) - Semantic BERT embeddings
                    'visual': (batch_size, 768) - Visual ViT embeddings
                    'acoustic': (2[list], batch_size, n_tones) - Acoustic tone labels
                subj_id: (batch_size, n_subjects) - Subject IDs
                token_mask: (batch_size, token_len) - Token mask for acoustic task

        Returns:
            outputs: dict - Predictions for each task
            loss: DotDict - Individual and total losses
        """
        # Initialize components of inputs
        X = inputs[0]
        targets_dict = inputs[1]
        subj_id = inputs[2]
        token_mask = inputs[3] if len(inputs) > 3 else None

        # ===== Shared encoder forward =====
        # Forward subject block to get subject-transformed signals
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))

        # Forward tokenizer to get tokenized tokens
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape

        # Reshape tokens to get init embedding
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))

        # Forward encoder block to get time-aligned token sequence
        E = self.encoder(E)

        # ===== Task-specific forward =====
        outputs = {}
        losses = {}

        # ----- Semantic Alignment Task -----
        if 'semantic' in targets_dict:
            Y_semantic = targets_dict['semantic']

            # Forward semantic alignment head
            Z_semantic = self.semantic_head(E)  # (batch_size, 768)

            # L2 normalization
            Z_semantic_norm = F.normalize(Z_semantic, p=2, dim=-1)
            Y_semantic_norm = F.normalize(Y_semantic, p=2, dim=-1)

            # Contrastive loss on encoder embeddings
            E_vq, loss_vq, _ = self.vq_block(E)
            E_norm = F.normalize(E, p=2, dim=-1)
            loss_contra_semantic, _ = self.contra_block(((E_norm, E_norm), (Y_semantic_norm, Y_semantic_norm)))

            # Alignment loss
            loss_align_semantic = self._loss_align(Z_semantic_norm, Y_semantic_norm)

            # Total semantic loss
            loss_semantic = (
                self.params.semantic_align_loss_scale * loss_align_semantic +
                self.params.semantic_contra_loss_scale * loss_contra_semantic
            )

            outputs['semantic'] = Z_semantic_norm
            losses['semantic_align'] = loss_align_semantic
            losses['semantic_contra'] = loss_contra_semantic
            losses['semantic'] = loss_semantic

        # ----- Visual Alignment Task -----
        if 'visual' in targets_dict:
            Y_visual = targets_dict['visual']

            # Apply task-specific mapping with residual connection if enabled
            E_visual = self.visual_mapping(E) + E if hasattr(self, 'visual_mapping') else E

            # Forward visual alignment head
            Z_visual = self.visual_head(E_visual)  # (batch_size, 768)

            # L2 normalization
            Z_visual_norm = F.normalize(Z_visual, p=2, dim=-1)
            Y_visual_norm = F.normalize(Y_visual, p=2, dim=-1)

            # Contrastive loss on encoder embeddings
            if 'semantic' not in targets_dict:  # Only compute VQ once
                E_vq, loss_vq, _ = self.vq_block(E)
            E_norm = F.normalize(E, p=2, dim=-1)
            loss_contra_visual, _ = self.contra_block(((E_norm, E_norm), (Y_visual_norm, Y_visual_norm)))

            # Alignment loss
            loss_align_visual = self._loss_align(Z_visual_norm, Y_visual_norm)

            # Total visual loss
            loss_visual = (
                self.params.visual_align_loss_scale * loss_align_visual +
                self.params.visual_contra_loss_scale * loss_contra_visual
            )

            outputs['visual'] = Z_visual_norm
            losses['visual_align'] = loss_align_visual
            losses['visual_contra'] = loss_contra_visual
            losses['visual'] = loss_visual

        # ----- Acoustic Classification Task -----
        if 'acoustic' in targets_dict:
            t_true = targets_dict['acoustic']  # (2[list], batch_size, n_tones)

            # Apply task-specific mapping with residual connection if enabled
            E_acoustic = self.acoustic_mapping(E) + E if hasattr(self, 'acoustic_mapping') else E

            # Forward acoustic classification heads
            t_pred = [self.acoustic_heads[tone_idx](E_acoustic) for tone_idx in range(len(self.acoustic_heads))]
            # t_pred - (2[list], batch_size, token_len, n_tones)

            # Expand t_true to match per-token supervision
            token_len = t_pred[0].shape[1]
            t_true_expanded = [t_true_i.unsqueeze(1).expand(-1, token_len, -1) for t_true_i in t_true]

            # Calculate classification loss
            weight = token_mask.to(dtype=t_pred[0].dtype) if token_mask is not None else None
            loss_cls_list = [self._loss_cls(t_pred_i, t_true_expanded_i, weight=weight)\
                for t_pred_i, t_true_expanded_i in zip(t_pred, t_true_expanded)]
            loss_cls_acoustic = torch.mean(torch.stack(loss_cls_list, dim=0))

            # Optional contrastive loss for acoustic task
            if self.params.acoustic_use_contra:
                # Use acoustic tone embeddings as targets for contrastive learning
                if 'semantic' not in targets_dict and 'visual' not in targets_dict:
                    E_vq, loss_vq, _ = self.vq_block(E)
                E_norm = F.normalize(E, p=2, dim=-1)
                # For acoustic, we use encoder self-similarity as the contrastive target
                loss_contra_acoustic, _ = self.contra_block(((E_norm, E_norm), (E_norm, E_norm)))
                loss_acoustic = (
                    self.params.acoustic_cls_loss_scale * loss_cls_acoustic +
                    self.params.acoustic_contra_loss_scale * loss_contra_acoustic
                )
                losses['acoustic_contra'] = loss_contra_acoustic
            else:
                loss_acoustic = self.params.acoustic_cls_loss_scale * loss_cls_acoustic

            outputs['acoustic'] = t_pred
            losses['acoustic_cls'] = loss_cls_acoustic
            losses['acoustic_cls_tone1'] = loss_cls_list[0]
            losses['acoustic_cls_tone2'] = loss_cls_list[1]
            losses['acoustic'] = loss_acoustic

        # ===== Combine multi-task losses =====
        if self.params.use_uncertainty_weighting:
            # Uncertainty-based automatic task weighting (Kendall et al., 2018)
            # loss_weighted = (1 / (2 * sigma^2)) * loss + log(sigma)
            # where sigma^2 = exp(log_var)
            loss_total = 0.0
            if 'semantic' in losses:
                precision_semantic = torch.exp(-self.log_var_semantic)
                loss_total += precision_semantic * losses['semantic'] + self.log_var_semantic
            if 'visual' in losses:
                precision_visual = torch.exp(-self.log_var_visual)
                loss_total += precision_visual * losses['visual'] + self.log_var_visual
            if 'acoustic' in losses:
                precision_acoustic = torch.exp(-self.log_var_acoustic)
                loss_total += precision_acoustic * losses['acoustic'] + self.log_var_acoustic
        else:
            # Manual task weighting
            loss_total = 0.0
            if 'semantic' in losses:
                loss_total += self.params.task_weight_semantic * losses['semantic']
            if 'visual' in losses:
                loss_total += self.params.task_weight_visual * losses['visual']
            if 'acoustic' in losses:
                loss_total += self.params.task_weight_acoustic * losses['acoustic']

        losses['total'] = loss_total

        # Convert to DotDict for consistency with other models
        loss = DotDict(losses)

        # Return outputs and losses
        return outputs, loss

    # def extract_embeddings func
    def extract_embeddings(self, inputs):
        """
        Extract embeddings from all three tasks for fusion classifier.

        Args:
            inputs: tuple - The input data [X, subj_id]
                X: (batch_size, seq_len, n_channels) - Brain signals
                subj_id: (batch_size, n_subjects) - Subject IDs

        Returns:
            fused_emb: (batch_size, d_fusion) - Concatenated embeddings from all three tasks
            emb_dict: dict - Individual embeddings for each task
        """
        X = inputs[0]
        subj_id = inputs[1]

        # Forward shared encoder
        X_h = self.subj_block((X, subj_id))
        T = self.tokenizer(X_h); token_shape = T.shape
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        E = self.encoder(E)

        # Extract embeddings from each task head
        # Semantic - (batch_size, 768)
        semantic_emb = self.semantic_head(E)
        semantic_emb = F.normalize(semantic_emb, p=2, dim=-1)

        # Visual - (batch_size, 768)
        visual_emb = self.visual_head(E)
        visual_emb = F.normalize(visual_emb, p=2, dim=-1)

        # Acoustic - Extract features before final classification
        # acoustic_emb - (batch_size, d_hidden[-1])
        acoustic_features_list = []
        for acoustic_head in self.acoustic_heads:
            features, _ = acoustic_head(E, return_features=True)
            acoustic_features_list.append(features)
        # Average features from both tone heads
        acoustic_emb = torch.mean(torch.stack(acoustic_features_list, dim=0), dim=0)
        # acoustic_emb - (batch_size, d_hidden[-1] or d_model)

        # Concatenate embeddings
        fused_emb = torch.cat([semantic_emb, visual_emb, acoustic_emb], dim=-1)
        # fused_emb - (batch_size, 768 + 768 + d_acoustic)

        emb_dict = {
            'semantic': semantic_emb,
            'visual': visual_emb,
            'acoustic': acoustic_emb,
            'fused': fused_emb
        }

        return fused_emb, emb_dict

    """
    loss funcs
    """
    # def _loss_align func
    def _loss_align(self, value, target):
        """
        Calculates alignment loss (MSE) between predicted and target embeddings.

        Args:
            value: (batch_size, d_output) - Predicted embedding from the model.
            target: (batch_size, d_output) - Ground-truth or teacher embedding.

        Returns:
            loss: torch.float32 - Mean squared error between L2-normalized embeddings.
        """
        assert value.shape == target.shape, f"Shape mismatch: {value.shape} vs {target.shape}"
        loss = 1000 * F.mse_loss(value, target, reduction="mean")
        return loss

    # def _loss_cls func
    def _loss_cls(self, value, target, weight=None):
        """
        Calculates classification loss between tensors value and target.

        Args:
            value: (batch_size, emb_len, n_tones) - Value of the object.
            target: (batch_size, emb_len, n_tones) - Target of the object.
            weight: (batch_size, emb_len) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        batch_size, emb_len, n_tones = value.shape

        # Apply L2 normalization to logits if enabled
        if self.params.acoustic_cls.use_l2_norm:
            value_normalized = F.normalize(value, p=2, dim=-1, eps=1e-12)
        else:
            value_normalized = value

        # Calculate cross-entropy loss
        loss = torch.reshape(F.cross_entropy(
            input=torch.reshape(value_normalized, shape=(-1, n_tones)),
            target=torch.reshape(target, shape=(-1, n_tones)),
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="none", label_smoothing=0.
        ), shape=(batch_size, emb_len))

        # Weight loss according to weight
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)

        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

# def duin_fusion_cls class
class duin_fusion_cls(nn.Module):
    """
    DuIN fusion classifier for 61-word classification.
    Uses pretrained multi-task model to extract embeddings from semantic, visual, and acoustic tasks,
    then fuses them for end-to-end 61-word classification.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_fusion_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_fusion_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        super(duin_fusion_cls, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Load pretrained multi-task model
        # Use params from self.params which contains architecture from pretrained checkpoint
        # (copied in run_fusion_cls.py from the actual checkpoint weights)
        # DO NOT create new duin_multitask_params here as it will reset to default values!

        # Initialize the multi-task model using the params from pretrained checkpoint
        # self.params already contains all the necessary multitask model parameters
        # (n_subjects, n_channels, seq_len, encoder, tokenizer, etc.) copied from the
        # pretrained checkpoint in run_fusion_cls.py
        self.multitask_model = duin_multitask(params=self.params)

        # Freeze encoder if specified
        if self.params.freeze_encoder:
            for param in self.multitask_model.subj_block.parameters():
                param.requires_grad = False
            for param in self.multitask_model.tokenizer.parameters():
                param.requires_grad = False
            for param in self.multitask_model.encoder.parameters():
                param.requires_grad = False
            print("INFO: Frozen encoder (SubjectBlock + Tokenizer + Encoder)")

        # Freeze task heads if specified
        if self.params.freeze_task_heads:
            for param in self.multitask_model.semantic_head.parameters():
                param.requires_grad = False
            for param in self.multitask_model.visual_head.parameters():
                param.requires_grad = False
            for param in self.multitask_model.acoustic_heads.parameters():
                param.requires_grad = False
            print("INFO: Frozen task heads (Semantic + Visual + Acoustic)")

        # Calculate fusion dimension dynamically from loaded multitask model
        # d_fusion = 768 (semantic) + 768 (visual) + d_acoustic
        # d_acoustic depends on the acoustic head's d_hidden[-1] or d_model
        acoustic_d_hidden = self.params.acoustic_cls.d_hidden
        d_acoustic = acoustic_d_hidden[-1] if len(acoustic_d_hidden) > 0 else self.params.encoder.d_model
        self.d_fusion = 768 + 768 + d_acoustic

        print(f"INFO: Fusion dimension calculated as {self.d_fusion} (768 + 768 + {d_acoustic})")

        # Initialize fusion classification head
        # fusion_head - (batch_size, d_fusion) -> (batch_size, n_labels)
        self.fusion_head = nn.Sequential()

        # Add hidden layers
        for hidden_idx in range(len(self.params.fusion.d_hidden)):
            self.fusion_head.append(nn.Sequential(
                nn.Linear(
                    in_features=(self.params.fusion.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.d_fusion),
                    out_features=self.params.fusion.d_hidden[hidden_idx],
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))

        # Add dropout if specified
        if self.params.fusion.dropout > 0.:
            self.fusion_head.append(nn.Dropout(p=self.params.fusion.dropout, inplace=False))

        # Add final classification layer (raw logits for cross-entropy loss)
        self.fusion_head.append(
            nn.Linear(
                in_features=(self.params.fusion.d_hidden[-1] if len(self.params.fusion.d_hidden) > 0 else self.d_fusion),
                out_features=self.params.fusion.n_labels,
                bias=True, device=None, dtype=None
            )
        )

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for fusion head only (multitask model weights are loaded separately)
        for module_i in self.fusion_head.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified multi-task checkpoint path.

        Args:
            path_ckpt: str - The path of the specified multi-task checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(path_ckpt, map_location=device)

        # Load weights into multitask_model
        # The checkpoint should contain all multitask model parameters
        model_dict = {}
        for parameter_name_i in ckpt_dict.keys():
            # Add "multitask_model." prefix to all checkpoint keys
            model_dict[f"multitask_model.{parameter_name_i}"] = ckpt_dict[parameter_name_i]

        # Load the state dict (strict=False to allow fusion_head to be uninitialized)
        missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)

        # Log information related to parameter load.
        print((
            "INFO: Complete loading pretrained multi-task weights from checkpoint ({}) in models.duin.duin_fusion_cls."
        ).format(path_ckpt))
        print(f"INFO: Missing keys (fusion_head): {[k for k in missing_keys if 'fusion_head' in k]}")
        print(f"INFO: Unexpected keys: {unexpected_keys}")

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_fusion_cls` to get 61-word classification predictions.

        Args:
            inputs: tuple - The input data [X, y_true, subj_id]
                X: (batch_size, seq_len, n_channels) - Brain signals
                y_true: (batch_size,) - Ground-truth word labels (for loss computation)
                subj_id: (batch_size, n_subjects) - Subject IDs

        Returns:
            y_pred: (batch_size, 61) - Predicted word probabilities
            loss: DotDict - Classification loss
        """
        # Initialize components of inputs
        X = inputs[0]
        y_true = inputs[1]
        subj_id = inputs[2]

        # Extract fused embeddings from multi-task model
        # If encoder is frozen, use torch.no_grad for efficiency
        if self.params.freeze_encoder and self.params.freeze_task_heads:
            with torch.no_grad():
                fused_emb, emb_dict = self.multitask_model.extract_embeddings((X, subj_id))
        else:
            fused_emb, emb_dict = self.multitask_model.extract_embeddings((X, subj_id))

        # Forward through fusion classification head
        y_pred = self.fusion_head(fused_emb)  # (batch_size, 61)

        # Calculate classification loss
        loss_cls = self._loss_cls(y_pred, y_true)

        # Calculate total loss
        loss_total = self.params.cls_loss_scale * loss_cls

        # Prepare loss dict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })

        # Return predictions and loss
        return y_pred, loss

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between prediction and target.

        Args:
            value: (batch_size, n_labels) - Predicted probabilities.
            target: (batch_size, n_labels) - Target one-hot labels.

        Returns:
            loss: torch.float32 - Cross-entropy loss.
        """
        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            input=value, target=target,
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.multitask_model.subj_block.get_weight_i()

# def duin_joint_multitask_cls class
class duin_joint_multitask_cls(nn.Module):
    """
    DuIN joint multi-task + end-to-end classification model.
    Combines multi-task learning (semantic/visual/acoustic) with direct 61-word classification.

    Architecture:
    - Shared encoder (SubjectBlock → Tokenizer → TimeEmbedding → Encoder)
    - 3 task heads for multi-task losses (semantic, visual, acoustic)
    - Fusion MLP for end-to-end classification
    - Trains from MAE checkpoint with all 4 losses combined
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_joint_multitask_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_joint_multitask_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        super(duin_joint_multitask_cls, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # ===== Initialize shared encoder components (same as duin_multitask) =====
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )

        # ===== Initialize VQ block (used for contrastive learning in alignment tasks) =====
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )

        # ===== Initialize contrastive block (shared by alignment tasks) =====
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)

        # ===== Initialize task-specific heads (same as duin_multitask) =====
        # 1. Semantic alignment head - (batch_size, emb_len, d_model) -> (batch_size, 768)
        self.semantic_head = AlignHead(params=self.params.semantic_align)

        # 2. Visual alignment head - (batch_size, emb_len, d_model) -> (batch_size, 768)
        self.visual_head = AlignHead(params=self.params.visual_align)

        # 3. Acoustic classification heads - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_tones)
        cls_tone1_params = cp.deepcopy(self.params.acoustic_cls); cls_tone1_params.n_tokens = cls_tone1_params.n_tone1
        cls_tone2_params = cp.deepcopy(self.params.acoustic_cls); cls_tone2_params.n_tokens = cls_tone2_params.n_tone2
        self.acoustic_heads = nn.ModuleList(modules=[
            TokenCLSHead(params=cls_tone1_params),
            TokenCLSHead(params=cls_tone2_params),
        ])

        # ===== Task-specific mapping matrices with residual connection =====
        if self.params.get('use_task_mapping', False):
            self.semantic_mapping = nn.Linear(
                self.params.encoder.d_model,
                self.params.encoder.d_model
            )
            self.visual_mapping = nn.Linear(
                self.params.encoder.d_model,
                self.params.encoder.d_model
            )
            self.acoustic_mapping = nn.Linear(
                self.params.encoder.d_model,
                self.params.encoder.d_model
            )
            print(f"INFO: Task-specific mapping matrices enabled (d_model={self.params.encoder.d_model})")

        # ===== Calculate fusion dimension dynamically =====
        # d_fusion = 768 (semantic) + 768 (visual) + d_acoustic
        acoustic_d_hidden = self.params.acoustic_cls.d_hidden
        d_acoustic = acoustic_d_hidden[-1] if len(acoustic_d_hidden) > 0 else self.params.encoder.d_model
        self.d_fusion = 768 + 768 + d_acoustic

        print(f"INFO: Fusion dimension calculated as {self.d_fusion} (768 + 768 + {d_acoustic})")

        # ===== Initialize fusion classification head (same as duin_fusion_cls) =====
        # fusion_head - (batch_size, d_fusion) -> (batch_size, n_labels)
        self.fusion_head = nn.Sequential()

        # Add hidden layers
        for hidden_idx in range(len(self.params.fusion.d_hidden)):
            self.fusion_head.append(nn.Sequential(
                nn.Linear(
                    in_features=(self.params.fusion.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.d_fusion),
                    out_features=self.params.fusion.d_hidden[hidden_idx],
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))

        # Add dropout if specified
        if self.params.fusion.dropout > 0.:
            self.fusion_head.append(nn.Dropout(p=self.params.fusion.dropout, inplace=False))

        # Add final classification layer (raw logits for cross-entropy loss)
        self.fusion_head.append(
            nn.Linear(
                in_features=(self.params.fusion.d_hidden[-1] if len(self.params.fusion.d_hidden) > 0 else self.d_fusion),
                out_features=self.params.fusion.n_labels,
                bias=True, device=None, dtype=None
            )
        )

        # ===== Initialize learnable task weights for uncertainty-based multi-task learning (optional) =====
        if self.params.use_uncertainty_weighting:
            # Log-variance parameters for automatic task balancing (Kendall et al., 2018)
            self.log_var_semantic = nn.Parameter(torch.zeros(1))
            self.log_var_visual = nn.Parameter(torch.zeros(1))
            self.log_var_acoustic = nn.Parameter(torch.zeros(1))
            self.log_var_e2e = nn.Parameter(torch.zeros(1))  # NEW: for e2e classification task

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for all components (encoder will be loaded from MAE checkpoint)
        # Task heads will be randomly initialized
        # Fusion head will be randomly initialized
        for module_i in self.fusion_head.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

        # Initialize task-specific mapping matrices if enabled
        if hasattr(self, 'semantic_mapping'):
            nn.init.xavier_uniform_(self.semantic_mapping.weight)
            nn.init.zeros_(self.semantic_mapping.bias)
        if hasattr(self, 'visual_mapping'):
            nn.init.xavier_uniform_(self.visual_mapping.weight)
            nn.init.zeros_(self.visual_mapping.bias)
        if hasattr(self, 'acoustic_mapping'):
            nn.init.xavier_uniform_(self.acoustic_mapping.weight)
            nn.init.zeros_(self.acoustic_mapping.bias)

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from MAE checkpoint (encoder only).

        Args:
            path_ckpt: str - The path of the specified MAE checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(path_ckpt, map_location=device)

        # Load encoder components only (same as duin_multitask)
        model_dict = {}
        module_map = {
            "([^.]*\\.)*subj_block": "subj_block",
            "([^.]*\\.)*tokenizer": "tokenizer",
            "([^.]*\\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]
                    break

        # Load the state dict (strict=False to allow task heads and fusion head to be uninitialized)
        self.load_state_dict(model_dict, strict=False)

        # Log information related to parameter load.
        print((
            "INFO: Complete loading pretrained MAE encoder weights from checkpoint ({}) in models.duin.duin_joint_multitask_cls."
        ).format(path_ckpt))
        print("INFO: Task heads (semantic, visual, acoustic) and fusion head initialized randomly.")

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_joint_multitask_cls` to compute both multitask losses and e2e classification loss.

        Args:
            inputs: tuple - The input data [X, targets_dict, subj_id, token_mask]
                X: (batch_size, seq_len, n_channels) - Brain signals
                targets_dict: dict - Ground-truth targets for each task
                    'semantic': (batch_size, 768) - Semantic BERT embeddings
                    'visual': (batch_size, 768) - Visual ViT embeddings
                    'acoustic': (2[list], batch_size, n_tones) - Acoustic tone labels
                    'y_true': (batch_size,) - Word labels for e2e classification (NEW!)
                subj_id: (batch_size, n_subjects) - Subject IDs
                token_mask: (batch_size, token_len) - Token mask for acoustic task

        Returns:
            outputs: dict - Predictions for each task
            loss: DotDict - Individual and total losses
        """
        # Initialize components of inputs
        X = inputs[0]
        targets_dict = inputs[1]
        subj_id = inputs[2]
        token_mask = inputs[3] if len(inputs) > 3 else None

        # ===== Shared encoder forward =====
        # Forward subject block to get subject-transformed signals
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))

        # Forward tokenizer to get tokenized tokens
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape

        # Reshape tokens to get init embedding
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))

        # Forward encoder block to get time-aligned token sequence
        E = self.encoder(E)

        # ===== Task-specific forward (same as duin_multitask) =====
        outputs = {}
        losses = {}

        # ----- Semantic Alignment Task -----
        if 'semantic' in targets_dict:
            Y_semantic = targets_dict['semantic']

            # Apply task-specific mapping with residual connection if enabled
            E_semantic = self.semantic_mapping(E) + E if hasattr(self, 'semantic_mapping') else E

            # Forward semantic alignment head
            Z_semantic = self.semantic_head(E_semantic)  # (batch_size, 768)

            # L2 normalization
            Z_semantic_norm = F.normalize(Z_semantic, p=2, dim=-1)
            Y_semantic_norm = F.normalize(Y_semantic, p=2, dim=-1)

            # Contrastive loss on encoder embeddings
            E_vq, loss_vq, _ = self.vq_block(E)
            E_norm = F.normalize(E, p=2, dim=-1)
            loss_contra_semantic, _ = self.contra_block(((E_norm, E_norm), (Y_semantic_norm, Y_semantic_norm)))

            # Alignment loss
            loss_align_semantic = self._loss_align(Z_semantic_norm, Y_semantic_norm)

            # Total semantic loss
            loss_semantic = (
                self.params.semantic_align_loss_scale * loss_align_semantic +
                self.params.semantic_contra_loss_scale * loss_contra_semantic
            )

            outputs['semantic'] = Z_semantic_norm
            losses['semantic_align'] = loss_align_semantic
            losses['semantic_contra'] = loss_contra_semantic
            losses['semantic'] = loss_semantic

        # ----- Visual Alignment Task -----
        if 'visual' in targets_dict:
            Y_visual = targets_dict['visual']

            # Apply task-specific mapping with residual connection if enabled
            E_visual = self.visual_mapping(E) + E if hasattr(self, 'visual_mapping') else E

            # Forward visual alignment head
            Z_visual = self.visual_head(E_visual)  # (batch_size, 768)

            # L2 normalization
            Z_visual_norm = F.normalize(Z_visual, p=2, dim=-1)
            Y_visual_norm = F.normalize(Y_visual, p=2, dim=-1)

            # Contrastive loss on encoder embeddings
            if 'semantic' not in targets_dict:  # Only compute VQ once
                E_vq, loss_vq, _ = self.vq_block(E)
            E_norm = F.normalize(E, p=2, dim=-1)
            loss_contra_visual, _ = self.contra_block(((E_norm, E_norm), (Y_visual_norm, Y_visual_norm)))

            # Alignment loss
            loss_align_visual = self._loss_align(Z_visual_norm, Y_visual_norm)

            # Total visual loss
            loss_visual = (
                self.params.visual_align_loss_scale * loss_align_visual +
                self.params.visual_contra_loss_scale * loss_contra_visual
            )

            outputs['visual'] = Z_visual_norm
            losses['visual_align'] = loss_align_visual
            losses['visual_contra'] = loss_contra_visual
            losses['visual'] = loss_visual

        # ----- Acoustic Classification Task -----
        if 'acoustic' in targets_dict:
            t_true = targets_dict['acoustic']  # (2[list], batch_size, n_tones)

            # Apply task-specific mapping with residual connection if enabled
            E_acoustic = self.acoustic_mapping(E) + E if hasattr(self, 'acoustic_mapping') else E

            # Forward acoustic classification heads
            t_pred = [self.acoustic_heads[tone_idx](E_acoustic) for tone_idx in range(len(self.acoustic_heads))]
            # t_pred - (2[list], batch_size, token_len, n_tones)

            # Expand t_true to match per-token supervision
            token_len = t_pred[0].shape[1]
            t_true_expanded = [t_true_i.unsqueeze(1).expand(-1, token_len, -1) for t_true_i in t_true]

            # Calculate classification loss
            weight = token_mask.to(dtype=t_pred[0].dtype) if token_mask is not None else None
            loss_cls_list = [self._loss_cls_acoustic(t_pred_i, t_true_expanded_i, weight=weight)\
                for t_pred_i, t_true_expanded_i in zip(t_pred, t_true_expanded)]
            loss_cls_acoustic = torch.mean(torch.stack(loss_cls_list, dim=0))

            # Optional contrastive loss for acoustic task
            if self.params.acoustic_use_contra:
                # Use acoustic tone embeddings as targets for contrastive learning
                if 'semantic' not in targets_dict and 'visual' not in targets_dict:
                    E_vq, loss_vq, _ = self.vq_block(E)
                E_norm = F.normalize(E, p=2, dim=-1)
                # For acoustic, we use encoder self-similarity as the contrastive target
                loss_contra_acoustic, _ = self.contra_block(((E_norm, E_norm), (E_norm, E_norm)))
                loss_acoustic = (
                    self.params.acoustic_cls_loss_scale * loss_cls_acoustic +
                    self.params.acoustic_contra_loss_scale * loss_contra_acoustic
                )
                losses['acoustic_contra'] = loss_contra_acoustic
            else:
                loss_acoustic = self.params.acoustic_cls_loss_scale * loss_cls_acoustic

            outputs['acoustic'] = t_pred
            losses['acoustic_cls'] = loss_cls_acoustic
            losses['acoustic_cls_tone1'] = loss_cls_list[0]
            losses['acoustic_cls_tone2'] = loss_cls_list[1]
            losses['acoustic'] = loss_acoustic

        # ===== NEW: E2E Classification Task =====
        if 'y_true' in targets_dict:
            y_true = targets_dict['y_true']

            # Extract fused embeddings (same as duin_fusion_cls)
            # Apply task-specific mapping with residual connection if enabled
            E_semantic = self.semantic_mapping(E) + E if hasattr(self, 'semantic_mapping') else E
            E_visual = self.visual_mapping(E) + E if hasattr(self, 'visual_mapping') else E
            E_acoustic = self.acoustic_mapping(E) + E if hasattr(self, 'acoustic_mapping') else E

            # Semantic - (batch_size, 768)
            semantic_emb = self.semantic_head(E_semantic)
            semantic_emb = F.normalize(semantic_emb, p=2, dim=-1)

            # Visual - (batch_size, 768)
            visual_emb = self.visual_head(E_visual)
            visual_emb = F.normalize(visual_emb, p=2, dim=-1)

            # Acoustic - Extract features before final classification
            # acoustic_emb - (batch_size, d_hidden[-1])
            acoustic_features_list = []
            for acoustic_head in self.acoustic_heads:
                features, _ = acoustic_head(E_acoustic, return_features=True)
                acoustic_features_list.append(features)
            # Average features from both tone heads
            acoustic_emb = torch.mean(torch.stack(acoustic_features_list, dim=0), dim=0)
            # acoustic_emb - (batch_size, d_hidden[-1] or d_model)

            # Concatenate embeddings
            fused_emb = torch.cat([semantic_emb, visual_emb, acoustic_emb], dim=-1)
            # fused_emb - (batch_size, 768 + 768 + d_acoustic)

            # Forward through fusion classification head
            y_pred = self.fusion_head(fused_emb)  # (batch_size, n_labels)

            # Calculate classification loss
            loss_cls = self._loss_cls_e2e(y_pred, y_true)

            # Scaled e2e classification loss
            loss_e2e = self.params.cls_loss_scale * loss_cls

            outputs['y_pred'] = y_pred
            losses['cls'] = loss_cls
            losses['e2e'] = loss_e2e

        # ===== Combine all losses (3 multitask losses + e2e loss) =====
        if self.params.use_uncertainty_weighting:
            # Uncertainty-based automatic task weighting (Kendall et al., 2018)
            # loss_weighted = (1 / (2 * sigma^2)) * loss + log(sigma)
            # where sigma^2 = exp(log_var)
            loss_total = 0.0
            if 'semantic' in losses:
                precision_semantic = torch.exp(-self.log_var_semantic)
                loss_total += precision_semantic * losses['semantic'] + self.log_var_semantic
            if 'visual' in losses:
                precision_visual = torch.exp(-self.log_var_visual)
                loss_total += precision_visual * losses['visual'] + self.log_var_visual
            if 'acoustic' in losses:
                precision_acoustic = torch.exp(-self.log_var_acoustic)
                loss_total += precision_acoustic * losses['acoustic'] + self.log_var_acoustic
            if 'e2e' in losses:
                precision_e2e = torch.exp(-self.log_var_e2e)
                loss_total += precision_e2e * losses['e2e'] + self.log_var_e2e
        else:
            # Manual task weighting
            loss_total = 0.0
            if 'semantic' in losses:
                loss_total += self.params.task_weight_semantic * losses['semantic']
            if 'visual' in losses:
                loss_total += self.params.task_weight_visual * losses['visual']
            if 'acoustic' in losses:
                loss_total += self.params.task_weight_acoustic * losses['acoustic']
            if 'e2e' in losses:
                loss_total += self.params.task_weight_e2e * losses['e2e']

        losses['total'] = loss_total

        # Convert to DotDict for consistency with other models
        loss = DotDict(losses)

        # Return outputs and losses
        return outputs, loss

    """
    loss funcs
    """
    # def _loss_align func
    def _loss_align(self, value, target):
        """
        Calculates alignment loss (MSE) between predicted and target embeddings.

        Args:
            value: (batch_size, d_output) - Predicted embedding from the model.
            target: (batch_size, d_output) - Ground-truth or teacher embedding.

        Returns:
            loss: torch.float32 - Mean squared error between L2-normalized embeddings.
        """
        assert value.shape == target.shape, f"Shape mismatch: {value.shape} vs {target.shape}"
        loss = 1000 * F.mse_loss(value, target, reduction="mean")
        return loss

    # def _loss_cls_acoustic func
    def _loss_cls_acoustic(self, value, target, weight=None):
        """
        Calculates acoustic classification loss between tensors value and target.

        Args:
            value: (batch_size, emb_len, n_tones) - Value of the object.
            target: (batch_size, emb_len, n_tones) - Target of the object.
            weight: (batch_size, emb_len) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        batch_size, emb_len, n_tones = value.shape

        # Apply L2 normalization to logits if enabled
        if self.params.acoustic_cls.use_l2_norm:
            value_normalized = F.normalize(value, p=2, dim=-1, eps=1e-12)
        else:
            value_normalized = value

        # Calculate cross-entropy loss
        loss = torch.reshape(F.cross_entropy(
            input=torch.reshape(value_normalized, shape=(-1, n_tones)),
            target=torch.reshape(target, shape=(-1, n_tones)),
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="none", label_smoothing=0.
        ), shape=(batch_size, emb_len))

        # Weight loss according to weight
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        return loss

    # def _loss_cls_e2e func
    def _loss_cls_e2e(self, value, target):
        """
        Calculates end-to-end classification loss between prediction and target.

        Args:
            value: (batch_size, n_labels) - Predicted probabilities.
            target: (batch_size,) or (batch_size, n_labels) - Target labels (int or one-hot).

        Returns:
            loss: torch.float32 - Cross-entropy loss.
        """
        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            input=value, target=target,
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

if __name__ == "__main__":
    import numpy as np
    # local dep
    import utils.model.torch
    from params.duin_params import duin_vqvae_params, duin_mae_params, duin_cls_params, duin_llm_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 3000; n_channels = 16; n_labels = 61; n_subjects = 10; d_llm = 1024

    # Initialize training process.
    utils.model.torch.set_seeds(42)

    ## Forward duin_vqvae.
    # Instantiate params.
    duin_vqvae_params_inst = duin_vqvae_params(dataset=dataset)
    duin_vqvae_params_inst.model.n_subjects = n_subjects
    duin_vqvae_params_inst.model.desubj.n_subjects = duin_vqvae_params_inst.model.subj.n_subjects = n_subjects
    duin_vqvae_params_inst.model.n_channels = n_channels
    duin_vqvae_params_inst.model.desubj.d_output = duin_vqvae_params_inst.model.subj.d_input = n_channels
    assert seq_len % duin_vqvae_params_inst.model.seg_len == 0; duin_vqvae_params_inst.model.seq_len = seq_len
    token_len = duin_vqvae_params_inst.model.seq_len // duin_vqvae_params_inst.model.tokenizer.seg_len
    duin_vqvae_params_inst.model.tokenizer.token_len = token_len
    duin_vqvae_params_inst.model.decoder.emb_len = duin_vqvae_params_inst.model.encoder.emb_len = token_len
    # Initialize input `X` & `subj_id` & `channel_mask`.
    # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    channel_mask = torch.ones((batch_size, n_channels), dtype=torch.bool)
    # Instantiate duin_vqvae.
    duin_vqvae_inst = duin_vqvae(duin_vqvae_params_inst.model); print(duin_vqvae_inst)
    # Forward layers in `duin_vqvae_inst`.
    # X_reconstr - (batch_size, seq_len, n_channels); loss - torch.float32
    X_reconstr, loss = duin_vqvae_inst((X, subj_id, channel_mask))
    # Forward layers before vector-quantizer in `duin_vqvae_inst`.
    # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
    E_vq, loss_vq, codex_probs = duin_vqvae_inst.quantize((X, subj_id))
    ## Forward duin_mae.
    # Instantiate params.
    duin_mae_params_inst = duin_mae_params(dataset=dataset)
    duin_mae_params_inst.model.subj.n_subjects = duin_mae_params_inst.model.n_subjects = n_subjects
    duin_mae_params_inst.model.subj.d_input = duin_mae_params_inst.model.n_channels = n_channels
    assert seq_len % duin_mae_params_inst.model.seg_len == 0; duin_mae_params_inst.model.seq_len = seq_len
    token_len = duin_mae_params_inst.model.seq_len // duin_mae_params_inst.model.tokenizer.seg_len
    duin_mae_params_inst.model.encoder.emb_len = duin_mae_params_inst.model.tokenizer.token_len = token_len
    # Initialize input `X` & `c_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
    emb_len = token_len; codex_size = duin_mae_params_inst.model.vq.codex_size
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    c_true = torch.tensor(np.eye(codex_size)[np.random.randint(0, codex_size, size=(batch_size, emb_len))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_mae.
    duin_mae_inst = duin_mae(duin_mae_params_inst.model); print(duin_mae_inst)
    # Forward layers in `duin_mae_inst`.
    # c_pred - (batch_size, emb_len, codex_size); loss - torch.float32
    c_pred, loss = duin_mae_inst((X, c_true, subj_id))
    ## Forward duin_cls.
    # Instantiate params.
    duin_cls_params_inst = duin_cls_params(dataset=dataset)
    duin_cls_params_inst.model.subj.n_subjects = duin_cls_params_inst.model.n_subjects = n_subjects
    duin_cls_params_inst.model.subj.d_input = duin_cls_params_inst.model.n_channels = n_channels
    assert seq_len % duin_cls_params_inst.model.seg_len == 0; duin_cls_params_inst.model.seq_len = seq_len
    token_len = duin_cls_params_inst.model.seq_len // duin_cls_params_inst.model.tokenizer.seg_len
    duin_cls_params_inst.model.tokenizer.token_len = token_len
    duin_cls_params_inst.model.encoder.emb_len = token_len
    duin_cls_params_inst.model.cls.d_feature = (
        duin_cls_params_inst.model.encoder.d_model * duin_cls_params_inst.model.encoder.emb_len
    )
    duin_cls_params_inst.model.cls.n_labels = n_labels
    # Initialize input `X` & `y_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_cls.
    duin_cls_inst = duin_cls(duin_cls_params_inst.model); print(duin_cls_inst)
    # Forward layers in `duin_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - torch.float32
    y_pred, loss = duin_cls_inst((X, y_true, subj_id))
    ## Forward duin_llm.
    # Instantiate params.
    duin_llm_params_inst = duin_llm_params(dataset=dataset)
    duin_llm_params_inst.model.subj.n_subjects = duin_llm_params_inst.model.n_subjects = n_subjects
    duin_llm_params_inst.model.subj.d_input = duin_llm_params_inst.model.n_channels = n_channels
    assert seq_len % duin_llm_params_inst.model.seg_len == 0; duin_llm_params_inst.model.seq_len = seq_len
    token_len = duin_llm_params_inst.model.seq_len // duin_llm_params_inst.model.tokenizer.seg_len
    duin_llm_params_inst.model.tokenizer.token_len = token_len
    duin_llm_params_inst.model.encoder.emb_len = token_len
    duin_llm_params_inst.model.rgs.d_model = duin_llm_params_inst.model.encoder.d_model
    duin_llm_params_inst.model.rgs.d_llm = d_llm
    # Initialize input `X` & `y_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); L - (batch_size, emb_len, d_llm); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    L = torch.rand((batch_size, token_len, d_llm), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_llm.
    duin_llm_inst = duin_llm(duin_llm_params_inst.model); print(duin_llm_inst)
    # Forward layers in `duin_llm_inst`.
    # L_rgs - (batch_size, emb_len, d_llm); loss - torch.float32
    L_rgs, loss = duin_llm_inst((X, L, subj_id))

