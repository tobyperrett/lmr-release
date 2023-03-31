#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import slowfast.utils.metrics as metrics

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
    "soft_target_cross_entropy": SoftTargetCrossEntropy,
}