"""
GradCAM class helps create localization maps using the Grad-CAM method for input image
and overlap the maps over the input images as heatmaps.
https://arxiv.org/pdf/1610.02391.pdf
"""
import torch
from pytorch_grad_cam import GradCAM as PyGradCam
import matplotlib.pyplot as plt

import fewpascal.datasets.utils as data_utils
from fewpascal.visualization.utils import get_layer


class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input image
    and overlap the maps over the input images as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
            self, model, target_layers, data_mean, data_std, colormap="viridis"
    ):
        self.model = model
        self.target_layer = get_layer(model, target_layers[0])
        self.reshape_transform = None
        self.data_mean = data_mean
        self.data_std = data_std
        self.colormap = plt.get_cmap(colormap)
        # TODO: Parametrise transformer versus others
        # TODO: Parametrise use cuda
        self.grad_cam = PyGradCam(
            model=model,
            target_layer=self.target_layer,
            use_cuda=True,
            reshape_transform=self.reshape_transform
        )

    def __call__(self, inputs, labels=None, alpha=0.5):
        inputs_clone = inputs.clone()
        preds = self.model(inputs_clone)
        localization_map = self.grad_cam(
            input_tensor=inputs,
            target_category=labels,
            eigen_smooth=True
        )

        if localization_map.device != torch.device("cpu"):
            localization_map = localization_map.cpu()
        heatmap = self.colormap(localization_map)
        inputs_clone = alpha * heatmap + (1 - alpha) * inputs_clone
        inputs_clone = data_utils.revert_tensor_normalize(
            inputs_clone, self.data_mean, self.data_std
        )

        return inputs_clone, preds

    @staticmethod
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(
            tensor.size(0), height, width, tensor.size(2)
        )

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
