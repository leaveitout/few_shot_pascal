#!/usr/bin/env python3

"""Head helper."""

import torch
import torch.nn as nn


class BasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
            self,
            cfg,
            dim_in,
    ):
        """
        Perform linear projection and activation as head for transformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
        """
        super().__init__()
        self.cfg = cfg
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.projection = nn.Linear(dim_in, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        act_func = cfg.MODEL.HEAD_ACT
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


class Embedding(nn.Module):
    """
    Embedding. No pool.
    """

    def __init__(
            self,
            cfg,
            dim_in,
    ):
        """
        Perform linear projection and activation as head for transformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
        """
        super().__init__()
        self.cfg = cfg
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.embedding_dim = cfg.MODEL.EMBEDDING_DIM
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.projection = nn.Linear(dim_in, self.embedding_dim, bias=True)

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        return x


class Similarity(nn.Module):
    """
    Embedding. No pool.
    """

    def __init__(
            self,
            cfg,
            dim_in,
    ):
        """
        Perform linear projection and activation as head for transformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
        """
        super().__init__()
        self.cfg = cfg

        # Softmax for evaluation and testing.
        act_func = cfg.MODEL.HEAD_ACT
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        else:
            raise AssertionError("Expected softmax activation")

    def forward(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        mat_mult = (
                text_features @ image_features.unsqueeze(-1)
        ).squeeze()

        similarity = (100.0 * mat_mult).softmax(dim=-1)
        return similarity
