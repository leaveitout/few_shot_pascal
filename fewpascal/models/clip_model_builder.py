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

    def __init__(self, model_name: str, jit: bool = True):
        """
        Init the backbone using clip.
        :param model_name: The name of the model
        :param jit: Use the jit version of the model
        """
        super().__init__()
        self.model, _ = clip.load(name=model_name, jit=jit)

    def forward(self, image: torch.Tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()

        return image_features

@MODEL_REGISTRY.register()
class FewShotClip(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self._backbone_name = cfg.MODEL.BACKBONE
        self._backbone = ClipImageBackbone(model_name=self._backbone_name)
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



# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
