import torch
import torch.nn as nn


def unfreeze_params(module: nn.Module):
    """
    makes params trainable
    :param module:
    :return:
    """
    for p in module.parameters():
        p.requires_grad = True


def freeze_params(module: nn.Module):
    """
    makes params untrainable
    :param module:
    :return:
    """
    for p in module.parameters():
        p.requires_grad = False

