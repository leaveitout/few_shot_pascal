#!/usr/bin/env python3

"""Clip models, see https://github.com/openai/CLIP for details"""

import torch
import clip

from . import head_helper
from .build import MODEL_REGISTRY


class ClipImageBackbone(torch.nn.Module):
    """
    Uses a clip model to get joint text image embeddings.
    """

    def __init__(self, model_name: str, vis_mode: bool = False):
        """
        Init the backbone using clip.
        :param model_name: The name of the model
        :param jit: Use the jit version of the model
        """
        super().__init__()
        self.vis_mode = vis_mode
        self.model, _ = clip.load(name=model_name, jit=not vis_mode)

    def forward(self, image: torch.Tensor):
        if self.vis_mode:
            image_features = self.model.encode_image(image).float()
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image).float()

        return image_features


@MODEL_REGISTRY.register()
class FewShotClip(torch.nn.Module):

    def __init__(self, cfg, vis_mode=False):
        super().__init__()

        self.cfg = cfg
        self._backbone_name = cfg.MODEL.BACKBONE
        self._backbone = ClipImageBackbone(model_name=self._backbone_name, vis_mode=vis_mode)
        self._head_name = cfg.MODEL.HEAD
        if self._head_name == "":
            self._head = torch.nn.Softmax(dim=-1)
        else:
            self._head = getattr(head_helper, self._head_name)(
                self.cfg,
                self._backbone.model.visual.proj.shape[-1]
            )

    def forward(self, image: torch.Tensor):
        image_features = self._backbone(image)

        probs = self._head(image_features)

        return probs


class ClipImageTextBackbone(torch.nn.Module):
    """
    Uses a clip model to get joint text image embeddings.
    """

    def __init__(self, model_name: str, vis_mode: bool = False):
        """
        Init the backbone using clip.
        :param model_name: The name of the model
        :param jit: Use the jit version of the model
        """
        super().__init__()
        self.vis_mode = vis_mode
        self.model, _ = clip.load(name=model_name, jit=not vis_mode)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        if self.vis_mode:
            image_features = self.model.encode_image(image).float()
            text_features = self.model.encode_text(text).float()
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image).float()
                batch_size, num_classes, latent_dim = text.shape
                text = text.view((batch_size * num_classes, latent_dim))
                text_features = self.model.encode_text(text).float()
                text_features = text_features.view(
                    (batch_size, num_classes, -1)
                )

        return image_features, text_features


@MODEL_REGISTRY.register()
class ZeroShotClip(torch.nn.Module):

    def __init__(self, cfg, vis_mode=False):
        super().__init__()

        self.cfg = cfg
        self._backbone_name = cfg.MODEL.BACKBONE
        self._backbone = ClipImageTextBackbone(model_name=self._backbone_name, vis_mode=vis_mode)
        self._head_name = cfg.MODEL.HEAD
        self._head = getattr(head_helper, self._head_name)(
            self.cfg,
            self._backbone.model.visual.proj.shape[-1]
        )

    def forward(self, image: torch.Tensor, texts: torch.Tensor):
        image_features, text_features = self._backbone(image, texts)

        probs = self._head(image_features, text_features)

        return probs
