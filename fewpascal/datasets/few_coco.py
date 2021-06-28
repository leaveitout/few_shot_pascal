#!/usr/bin/env python3
import random
from itertools import chain
from typing import Tuple, List
from pathlib import Path

import clip
import torch
import torch.utils.data
from fvcore.common.config import CfgNode
from torchvision import transforms

import fewpascal.utils.logging as logging

from . import utils
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, horizontal_flip

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Fewcoco(torch.utils.data.Dataset):
    """
    Dataset for the few-shot evaluation on coco images.
    """

    def __init__(self, cfg: CfgNode, mode: str) -> None:
        """
        Initialise the Few-shot coco dataset
        :param cfg: The configuration node.
        :param mode: One of 'train', 'test'
        """
        if mode not in ["train", "val", "test"]:
            raise AssertionError(f"Split {mode} not supported")
        self.mode = mode
        self.cfg = cfg

        # Multi-prompt testing
        # if self.mode in ["train", "val"]:
        #     self._num_prompts = 1
        if self.cfg.TOKENS.ENABLE and self.mode == "test":
            self._num_prompts = len(self.cfg.TOKENS.PROMPTS)
        else:
            self._num_prompts = 1

        # Multi-flip testing
        if self.mode in ["train", "val"] or not self.cfg.TEST.FLIPPING:
            self._num_flips = 1
        else:
            self._num_flips = 2

        # Multi-crop testing
        if self.mode in ["train", "val"]:
            self._num_crops = 1
        elif self.mode in ["test"]:
            self._num_crops = cfg.TEST.NUM_SPATIAL_CROPS

        self._num_repeats = self._num_prompts * self._num_flips * self._num_crops

        self._construct_loader()
        self.aug = False
        self.rand_erase = False

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self) -> None:
        """
        Construct the image loader
        """
        path_to_data = Path(self.cfg.DATA.PATH_TO_DATA_DIR)
        assert path_to_data.exists(), f"{path_to_data} does not exist."
        # TODO: Add validation, cross-validation
        path_to_split = path_to_data / self.mode
        if self.mode == "val":
            path_to_split = path_to_data / "test"

        assert path_to_split.exists(), f"{path_to_split} does not exist."

        self._label_idx_to_text = [
            p.name for p in path_to_split.iterdir() if p.is_dir()
        ]
        self._label_text_to_idx = {
            text: idx for idx, text in enumerate(self._label_idx_to_text)
        }

        self._image_paths = sorted(list(path_to_split.glob("*/*.jpg")))
        self._labels_text = [p.parent.parts[-1] for p in self._image_paths]
        # print(self._labels_text)
        self._labels_idxs = [
            self._label_text_to_idx[label] for label in self._labels_text
        ]

        # Repeat samples if we are taking more than 1 crop
        if self._num_repeats > 1:

            def chain_repeats(elements: List, num_repeats: int) -> List:
                return list(
                    chain.from_iterable([[el] * num_repeats for el in elements])
                )

            self._image_paths = chain_repeats(self._image_paths, self._num_repeats)
            self._labels_text = chain_repeats(self._labels_text, self._num_repeats)
            self._labels_idxs = chain_repeats(self._labels_idxs, self._num_repeats)

        # We need this to ensemble the crops.
        # self._spatial_idx = list(
        #     chain.from_iterable(
        #         [range(self._num_repeats) for _ in range(len(self._image_paths))]
        #     )
        # )

        logger.info(
            f"Few-shot COCO dataloader constructed " f"(size: {len(self._image_paths)})"
        )

    def __getitem__(self, index: int):
        flipping = False
        # crop_idx = -1 indicates random sampling
        # crop_idx is in [0, 1, 2]. Corresponding to left,
        # center, or right if width is larger than height, and top, middle,
        # or bottom if height is larger than width.
        crop_idx = -1
        prompt_idx = -1

        if self.mode == "test":
            flipping = bool(index % self._num_flips)

            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1:
                crop_idx = (index // self._num_flips) % self._num_crops
            else:
                # Central cropping
                crop_idx = 1

            prompt_idx = (
                index // self._num_flips // self._num_crops
            ) % self._num_prompts

            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        elif self.mode in ["train", "val"]:
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            raise AssertionError(f"{self.mode} is not a supported split.")

        image = utils.retry_load_images(
            [self._image_paths[index]], backend="pytorch", to_rgb=True
        )

        if self.aug:
            if self.cfg.AUG.NUM_SAMPLE > 1:
                image_list = []
                label_text_list = []
                label_idx_list = []
                index_list = []
                for _ in range(self.cfg.AUG.NUM_SAMPLE):
                    new_image = self._aug_frame(
                        image, crop_idx, min_scale, max_scale, crop_size,
                    )
                    label_text = None
                    if self.cfg.TOKENS.ENABLE:
                        label_text = self.tokenize_text(
                            self._labels_text[index], prompt_idx
                        )
                        label_text = label_text.squeeze(0)

                    label_idx = self._labels_idxs[index]
                    new_image = new_image.squeeze(0)

                    image_list.append(new_image)
                    label_text_list.append(label_text)
                    label_idx_list.append(label_idx)
                    index_list.append(index)

                if self.cfg.TOKENS.ENABLE:
                    return image_list, label_text_list, label_idx_list, index_list
                else:
                    return image_list, label_idx_list, index_list
            else:
                image = self._aug_image(
                    image, crop_idx, min_scale, max_scale, crop_size,
                )

        else:
            image = utils.tensor_normalize(image, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
            # N H W C -> N C H W.
            image = image.permute(0, 3, 1, 2)

            if flipping:
                image = horizontal_flip(1.0, image)

            # Perform data augmentation.
            image = utils.spatial_sampling(
                image,
                spatial_idx=crop_idx,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

        label_idx = self._labels_idxs[index]
        label_text = None
        if self.cfg.TOKENS.ENABLE:
            label_text = self.tokenize_text(self._labels_text[index], prompt_idx)
            label_text = label_text.squeeze(0)

        image = image.squeeze(0)

        if self.cfg.TOKENS.ENABLE:
            return image, label_text, label_idx, index
        else:
            return image, label_idx, index

    def _aug_image(
        self, images, spatial_sample_index, min_scale, max_scale, crop_size,
    ):
        """
        Augment the images

        :param images: Images tensor of shape (N, H, W, C).
        :param spatial_sample_index: As per spatial sampling
        :param min_scale: the minimal size of scaling.
        :param max_scale: the maximal size of scaling.
        :param crop_size: the size of height and width used to crop the frames.
        :return:
        """
        aug_transform = create_random_augment(
            input_size=(images.shape[-2], images.shape[-1]),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # N H W C -> N C H W
        images = images.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(images)
        list_img = aug_transform(list_img)
        images = self._list_img_to_frames(list_img)

        # Convert to channels-last, normalise, and convert back
        # N C H W -> N H W C
        images = images.permute(0, 2, 3, 1)
        images = utils.tensor_normalize(images, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # N H W C -> N C H W.
        images = images.permute(0, 3, 1, 2)

        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = None if (self.mode not in ["train"] or len(scl) == 0) else scl
        relative_aspect = None if (self.mode not in ["train"] or len(asp) == 0) else asp
        images = utils.spatial_sampling(
            images,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            images = images.permute(1, 0, 2, 3)
            images = erase_transform(images)
            images = images.permute(1, 0, 2, 3)

        return images

    @staticmethod
    def _frame_to_list_img(frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    @staticmethod
    def _list_img_to_frames(img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def tokenize_text(self, text: str, prompt_idx: int) -> torch.Tensor:
        """
        Tokenize text for model.

        :param text: The text to tokenize
        :param prompt_idx: The prompt to use, -1 indicates to use a random one.
        :return: Tokenized tensor
        """
        if prompt_idx < -1 or prompt_idx >= len(self.cfg.TOKENS.PROMPTS):
            raise AssertionError(f"prompt_idx {prompt_idx} is not valid.")

        if prompt_idx == -1:
            prompt_idx = random.randrange(len(self.cfg.TOKENS.PROMPTS))

        prompt = self.cfg.TOKENS.PROMPTS[prompt_idx].replace("{}", text)

        return clip.tokenize([prompt])

    def __len__(self) -> int:
        """
        Get the number of images in the dataset split.
        :return: The number of images in the dataset split.
        """
        return self.num_images

    @property
    def num_images(self) -> int:
        """
        Get the number of images in the dataset split.
        :return: The number of images in the dataset split.
        """
        return len(self._image_paths)
