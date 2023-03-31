#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""
import collections
from collections import OrderedDict
import copy
import math
import os
import random
import string
import torchvision
from torchvision.utils import make_grid, save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from . import vit_helper
from .build import MODEL_REGISTRY

import math

@MODEL_REGISTRY.register()
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.VIT.PATCH_SIZE
        self.in_chans = cfg.VIT.CHANNELS

        self.embed_dim = cfg.VIT.EMBED_DIM
        self.depth = cfg.VIT.DEPTH
        self.num_heads = cfg.VIT.NUM_HEADS
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.drop_path_rate = cfg.VIT.DROP_PATH
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.video_input = cfg.VIT.VIDEO_INPUT
        self.temporal_resolution = cfg.VIT.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.VIT.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.head_act = cfg.VIT.HEAD_ACT
        self.cfg = cfg

        self.device_param = nn.Parameter(torch.zeros(1))

        if cfg.TRAIN.DATASET == "Epickitchens":
            self.num_classes = [97, 300]
            if cfg.EPICKITCHENS.LT_TASK == "verb":
                self.lt_num_classes = 97
                train_counts = cfg.EPICKITCHENS.TRAIN_VERB_COUNTS
            elif cfg.EPICKITCHENS.LT_TASK == "noun":
                self.lt_num_classes = 300
                train_counts = cfg.EPICKITCHENS.TRAIN_NOUN_COUNTS
            else:
                raise NotImplementedError()

            fst_class_idxs = [i for i, e in enumerate(train_counts) if e <= cfg.TRAIN.RECONSTRUCT_FST_THRESHOLD]

            fst_classes = torch.zeros(self.lt_num_classes)
            for idx in fst_class_idxs:
                fst_classes[idx] = 1
            self.fst_classes = nn.Parameter(fst_classes, requires_grad=False)
            self.class_counts = nn.Parameter(torch.tensor(train_counts), requires_grad=False)

        elif cfg.TRAIN.DATASET == "Ssv2":
            self.num_classes = cfg.MODEL.NUM_CLASSES
            self.lt_num_classes = cfg.MODEL.NUM_CLASSES

            fst_class_idxs = [i for i, e in enumerate(cfg.SSV2.TRAIN_COUNTS) if e <= cfg.TRAIN.RECONSTRUCT_FST_THRESHOLD]

            fst_classes = torch.zeros(self.lt_num_classes)
            for idx in fst_class_idxs:
                fst_classes[idx] = 1
            self.fst_classes = nn.Parameter(fst_classes, requires_grad=False)
            self.class_counts = nn.Parameter(torch.tensor(cfg.SSV2.TRAIN_COUNTS), requires_grad=False)



        self.frozen_rep = False

        # Patch Embedding
        self.patch_embed = vit_helper.PatchEmbed(
            img_size=224, 
            patch_size=self.patch_size, 
            in_chans=self.in_chans, 
            embed_dim=self.embed_dim
        )

        # 3D Patch Embedding
        self.patch_embed_3d = vit_helper.PatchEmbed3D(
            img_size=self.img_size, 
            temporal_resolution=self.temporal_resolution, 
            patch_size=self.patch_size,
            in_chans=self.in_chans, 
            embed_dim=self.embed_dim, 
            z_block_size=self.cfg.VIT.PATCH_SIZE_TEMP
        )
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(
            self.patch_embed_3d.proj.weight.data)
        
        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.VIT.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=.02)

        if self.cfg.VIT.POS_EMBED == "joint":
            self.st_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim))
            trunc_normal_(self.st_embed, std=.02)
        elif self.cfg.VIT.POS_EMBED == "separate":
            self.temp_embed = nn.Parameter(
                torch.zeros(1, self.temporal_resolution, self.embed_dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        if self.cfg.VIT.ATTN_LAYER == "divided":
            self.blocks = nn.ModuleList([
                vit_helper.DividedSpaceTimeBlock(
                    attn_type=cfg.VIT.ATTN_LAYER, 
                    dim=self.embed_dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer, 
                )
                for i in range(self.depth - self.cfg.VIT.NUM_EXPERT_BLOCKS)
            ])
            self.expert_blocks = nn.ModuleList([
                nn.ModuleList([
                    vit_helper.DividedSpaceTimeBlock(
                        attn_type=cfg.VIT.ATTN_LAYER, 
                        dim=self.embed_dim, 
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio, 
                        qkv_bias=self.qkv_bias, 
                        drop=self.drop_rate, 
                        attn_drop=self.attn_drop_rate, 
                        drop_path=dpr[i], 
                        norm_layer=norm_layer, 
                    )
                    for i in range(self.cfg.VIT.NUM_EXPERT_BLOCKS)
                ])
                for j in range(self.cfg.VIT.NUM_EXPERTS)
            ])
        else:
            self.blocks = nn.ModuleList([
                vit_helper.Block(
                    attn_type=cfg.VIT.ATTN_LAYER, 
                    dim=self.embed_dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer,
                    use_original_code=self.cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE
                )
                for i in range(self.depth - self.cfg.VIT.NUM_EXPERT_BLOCKS)
            ])
            self.expert_blocks = nn.ModuleList([
                nn.ModuleList([
                    vit_helper.Block(
                        attn_type=cfg.VIT.ATTN_LAYER, 
                        dim=self.embed_dim, 
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio, 
                        qkv_bias=self.qkv_bias, 
                        drop=self.drop_rate, 
                        attn_drop=self.attn_drop_rate, 
                        drop_path=dpr[i], 
                        norm_layer=norm_layer,
                        use_original_code=self.cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE
                    )
                    for i in range(self.cfg.VIT.NUM_EXPERT_BLOCKS)
                ])
                for j in range(self.cfg.VIT.NUM_EXPERTS)
            ])
        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh() 
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                    ('act', act),
                ]))
                for i in range(self.cfg.VIT.NUM_EXPERTS)
            ])
        else:
            self.pre_logits = nn.ModuleList([nn.Identity() for i in range(self.cfg.VIT.NUM_EXPERTS)])

        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)

        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for h_idx, h_clss in enumerate(self.num_classes):
                n_head_clss = h_clss
                setattr(self, "head%d"%h_idx, nn.ModuleList([nn.Linear(self.embed_dim, n_head_clss) for i in range(self.cfg.VIT.NUM_EXPERTS)]))
        else:
            n_head_clss = self.num_classes
            self.head = nn.ModuleList([
                (nn.Linear(self.embed_dim, n_head_clss) 
                if self.num_classes > 0 else nn.Identity())
                for i in range(self.cfg.VIT.NUM_EXPERTS)
            ])

        if self.cfg.TRAIN.RECONSTRUCT_ATTENTION_TYPE == "q":
            self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        elif self.cfg.TRAIN.RECONSTRUCT_ATTENTION_TYPE == "q_k":
            self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

        if self.cfg.TRAIN.RECONSTRUCT_BANK_SIZE > 0:
            # one memory bank per expert
            self.mem_banks = [collections.deque(maxlen=self.cfg.TRAIN.RECONSTRUCT_BANK_SIZE) for i in range(self.cfg.VIT.NUM_EXPERTS)]
            self.label_banks = [collections.deque(maxlen=self.cfg.TRAIN.RECONSTRUCT_BANK_SIZE) for i in range(self.cfg.VIT.NUM_EXPERTS)]


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.VIT.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, global_pool=''):
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for h_idx, h_clss in enumerate(self.num_classes):
                cur_head = getattr(self, "head%d"%h_idx)
                for m in cur_head:
                    self._init_weights(m)

        else:
            for m in self.head:
                self._init_weights(m)

    def forward_features(self, x):
        if self.video_input:
            x = x[0]
        B = x.shape[0]

        # Tokenize input
        if self.cfg.VIT.PATCH_SIZE_TEMP > 1:
            x = self.patch_embed_3d(x)
        else:
            # 2D tokenization
            if self.video_input:
                x = x.permute(0, 2, 1, 3, 4)
                (B, T, C, H, W) = x.shape
                x = x.reshape(B*T, C, H, W)

            x = self.patch_embed(x)

            if self.video_input:
                (B2, T2, D2) = x.shape
                x = x.reshape(B, T*T2, D2)

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Interpolate positinoal embeddings
        if self.cfg.DATA.TRAIN_CROP_SIZE != 224:
            pos_embed = self.pos_embed
            N = pos_embed.shape[1] - 1
            npatch = int((x.size(1) - 1) / self.temporal_resolution)
            class_emb = pos_embed[:, 0]
            pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2),
                scale_factor=math.sqrt(npatch / N),
                mode='bicubic',
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            new_pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        else:
            new_pos_embed = self.pos_embed
            npatch = self.patch_embed.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.VIT.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1)
                tile_temporal_embed = self.temp_embed.repeat_interleave(
                    npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.VIT.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed
                            
        # Apply positional dropout
        x = self.pos_drop(x)

        # Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                seq_len=npatch,
                num_frames=self.temporal_resolution,
                approx=self.cfg.VIT.APPROX_ATTN_TYPE,
                num_landmarks=self.cfg.VIT.APPROX_ATTN_DIM
            )


        all_x_e = []
        # Expert encoding
        for e in range(self.cfg.VIT.NUM_EXPERTS):
            x_e = x
            for i, blk in enumerate(self.expert_blocks[e]):
                x_e = blk(
                    x_e,
                    seq_len=npatch,
                    num_frames=self.temporal_resolution,
                    approx=self.cfg.VIT.APPROX_ATTN_TYPE,
                    num_landmarks=self.cfg.VIT.APPROX_ATTN_DIM
                )
            x_e = self.norm(x_e)[:, 0]
            x_e = self.pre_logits[e](x_e)
            all_x_e.append(x_e)
        return all_x_e

    def forward(self, x, labels=None):

        if self.frozen_rep:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        x = [self.head_drop(i) for i in x]

        if self.cfg.TRAIN.RECONSTRUCT and self.training:

            if self.cfg.TRAIN.DATASET == "Epickitchens":
                if self.cfg.EPICKITCHENS.LT_TASK == "verb":
                    labels_orig = labels["verb"]
                elif self.cfg.EPICKITCHENS.LT_TASK == "noun":
                    labels_orig = labels["noun"]
                else:
                    raise NotImplementedError()
            else:
                labels_orig = labels


            if len(labels_orig.shape) > 1:
                label_idxs = torch.argmax(labels_orig, dim=1)
            else:
                label_idxs = labels_orig
            label_idxs = label_idxs.to(x[0].device)

            x_reconstructed = []
            for i in range(len(x)):

                q = x[i]
                q_labels = label_idxs

                if self.cfg.TRAIN.RECONSTRUCT_BANK_SIZE > 0:
                    self.mem_banks[i].append(x[i].detach())
                    self.label_banks[i].append(label_idxs.detach())
                    k = torch.cat(list(self.mem_banks[i]))
                    k_labels = torch.cat(list(self.label_banks[i]))
                else:
                    k = q
                    k_labels = q_labels

                q_is_fst_mask = torch.index_select(input=self.fst_classes, dim=0, index=q_labels)
                k_is_fst_mask = torch.index_select(input=self.fst_classes, dim=0, index=k_labels)

                if self.cfg.TRAIN.RECONSTRUCT_ATTENTION_TYPE == "cos_sim":
                    q_norm = F.normalize(input=q, p=1.0, dim=-1)
                    k_norm = F.normalize(input=k, p=1.0, dim=-1)
                    q_k_similarity = torch.matmul(q_norm, k_norm.t())
                elif self.cfg.TRAIN.RECONSTRUCT_ATTENTION_TYPE == "q":
                    q_norm = F.normalize(input=q, p=1.0, dim=-1)
                    k_norm = F.normalize(input=k, p=1.0, dim=-1)
                    q_norm = self.q_linear(q_norm)
                    k_norm = self.q_linear(k_norm)
                    q_k_similarity = torch.matmul(q_norm, k_norm.t())
                elif self.cfg.TRAIN.RECONSTRUCT_ATTENTION_TYPE == "q_k":
                    q_norm = F.normalize(input=q, p=1.0, dim=-1)
                    k_norm = F.normalize(input=k, p=1.0, dim=-1)
                    q_norm = self.q_linear(q_norm)
                    k_norm = self.k_linear(k_norm)
                    q_k_similarity = torch.matmul(q_norm, k_norm.t())
                else:
                    # static attention
                    q_norm = self.norm(q)
                    k_norm = self.norm(k)
                    q_k_similarity = torch.matmul(q_norm, k_norm.t()) / self.embed_dim

                # remove similarity to self
                mask = torch.eye(q_k_similarity.shape[0], q_k_similarity.shape[1], device=q_k_similarity.device)

                if self.cfg.TRAIN.RECONSTRUCT_CONTRIBUTION == "no-fst":
                    # don't allow fst classes to contribute to reconstruction
                    mask = mask + k_is_fst_mask
                    # set max mask value to 1
                    one_tensor = torch.tensor(1.0, device=mask.device, dtype=mask.dtype)
                    mask = torch.where(mask > one_tensor, one_tensor, mask)

                q_k_similarity = q_k_similarity - 1000 * mask
                q_k_similarity = F.softmax(q_k_similarity, dim=-1)

                reconstructed = torch.matmul(q_k_similarity, k)

                if self.cfg.TRAIN.RECONSTRUCT_APPLICATION == "log_params":
                    reconstruct_d = self.cfg.TRAIN.RECONSTRUCT_D
                    reconstruct_l = self.cfg.TRAIN.RECONSTRUCT_L
                    # The paper does not contain h for neatness, but it is a very small value that controls the contribution for the largest class.
                    # It would appear in eq. 6, and would only be significant if the largest class size in a dataset was small, which is not the case with long-tail data
                    reconstruct_h = self.cfg.TRAIN.RECONSTRUCT_H
                else:
                    raise NotImplementedError()

                epsilon = max(5, min(self.class_counts))
                class_counts_adjusted = 1 / torch.log((self.class_counts + epsilon) * reconstruct_d)
                class_weights = (class_counts_adjusted - torch.min(class_counts_adjusted))
                class_weights = class_weights / torch.max(class_weights) * (reconstruct_l - reconstruct_h) + reconstruct_h

                contrib = torch.index_select(input=class_weights, dim=0, index=q_labels)
                reconstructed = (reconstructed.t() * contrib).t() + (q.t() * (1 - contrib)).t()

                x_reconstructed.append(reconstructed)

            x = x_reconstructed


        if self.cfg.TRAIN.FEATURE_MIXUP_PROB > 0.0 and self.training:

            if self.cfg.TRAIN.DATASET == "Epickitchens":
                if self.cfg.EPICKITCHENS.LT_TASK == "verb":
                    labels_orig = labels["verb"]
                elif self.cfg.EPICKITCHENS.LT_TASK == "noun":
                    labels_orig = labels["noun"]
                else:
                    raise NotImplementedError()
            else:
                labels_orig = labels

            if len(labels_orig.shape) > 1:
                label_oh = labels_orig
            else:
                label_oh = torch.nn.functional.one_hot(labels_orig, self.lt_num_classes)
            label_oh = label_oh.float().to(x[0].device)

            samples_to_mixup = int(x[0].shape[0] * self.cfg.TRAIN.FEATURE_MIXUP_PROB)

            mixup_ids = torch.randint(low=0, high=x[0].shape[0], size=[x[0].shape[0]])
            mixup_ids = torch.nn.functional.one_hot(mixup_ids, x[0].shape[0]).to(x[0].device)
            mixup_weights = torch.rand([x[0].shape[0]], device=x[0].device).to(x[0].device)

            mixup_mat =  (torch.eye(x[0].shape[0], device=x[0].device) * (1 - mixup_weights) + mixup_ids.t() * mixup_weights).t()
            x = [torch.cat([torch.matmul(mixup_mat, e)[0:samples_to_mixup, :], e[samples_to_mixup:, :]], dim=0) for e in x]

            labels_mu = torch.cat([torch.matmul(mixup_mat, label_oh)[0:samples_to_mixup, :], label_oh[samples_to_mixup:, :]])

            if self.cfg.TRAIN.DATASET == "Epickitchens":
                if self.cfg.EPICKITCHENS.LT_TASK == "verb":
                    labels["verb"] = labels_mu
                elif self.cfg.EPICKITCHENS.LT_TASK == "noun":
                    labels["noun"] = labels_mu
            else:
                labels = labels_mu



        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            for head in range(len(self.num_classes)):
                experts = getattr(self, "head%d"%head)
                x_out = [experts[i](x[i]) for i in range(self.cfg.VIT.NUM_EXPERTS)]
                x_out_pred = torch.stack(x_out)
                x_out_pred = torch.sum(x_out_pred, dim=0)
                x_out_pred = torch.nn.functional.softmax(x_out_pred, dim=-1)
                if self.training:
                    x_out.insert(0, x_out_pred)
                else:
                    x_out = x_out_pred
                output.append(x_out)
            return output, labels
        else:
            experts = self.head
            x_out = [experts[i](x[i]) for i in range(self.cfg.VIT.NUM_EXPERTS)]
            x_out_pred = torch.stack(x_out)
            x_out_pred = torch.sum(x_out_pred, dim=0)
            x_out_pred = torch.nn.functional.softmax(x_out_pred, dim=-1)
            if self.training:
                x_out.insert(0, x_out_pred)
            else:
                x_out = x_out_pred
            return x_out, labels



    def freeze_representation(self):
        self.frozen_rep = True
