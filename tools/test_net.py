#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from iopath.common.file_io import g_pathmgr
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import fewpascal.utils.checkpoint as cu
import fewpascal.utils.distributed as du
import fewpascal.utils.logging as logging
import fewpascal.visualization.tensorboard_vis as tb
from fewpascal.datasets import loader
from fewpascal.models import build_model
from fewpascal.utils.meters import TestMeter
from fewpascal.datasets.build import build_dataset

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform multi-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            fewpascal/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, batch in enumerate(test_loader):
        if cfg.TOKENS.ENABLE:
            inputs, labels_text, labels_idxs, idxs = batch
        else:
            inputs, labels_idxs, idxs = batch
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            if cfg.TOKENS.ENABLE:
                labels_text = labels_text.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels_idxs = labels_idxs.cuda()
            idxs = idxs.cuda()

        test_meter.data_toc()

        # Perform the forward pass.
        if cfg.TOKENS.ENABLE:
            preds = model(inputs, labels_text)
        else:
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels_idxs, idxs])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels_idxs = labels_idxs.cpu()
            idxs = idxs.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds.detach(), labels_idxs.detach(), idxs.detach())
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    all_preds = test_meter.image_preds.clone().detach()
    all_labels = test_meter.image_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()
    if writer is not None:
        writer.plot_eval(preds=all_preds, labels=all_labels)

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        if du.is_root_proc():
            with g_pathmgr.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

        logger.info("Successfully saved prediction results to {}".format(save_path))

    test_meter.finalize_metrics()
    return test_meter


@torch.no_grad()
def perform_test_triplet(cfg, model):
    def get_all_embeddings(dataset, model):
        def get_data_and_label(data):
            return data[0], data[1]
        tester = testers.BaseTester(data_and_label_getter=get_data_and_label)
        return tester.get_all_embeddings(dataset, model)

    def test_triplet(train_set, test_set, model, accuracy_calculator):
        train_embeddings, train_labels = get_all_embeddings(train_set, model)
        test_embeddings, test_labels = get_all_embeddings(test_set, model)
        print("Computing accuracy")
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, train_embeddings, test_labels, train_labels, False
        )
        print(
            "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"])
        )

    # Construct the dataset
    split = "train"
    dataset_name = cfg.TRAIN.DATASET
    # batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
    # shuffle = True
    # drop_last = True
    train_set = build_dataset(dataset_name, cfg, split)

    split = "test"
    dataset_name = cfg.TEST.DATASET
    # batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
    # shuffle = False
    # drop_last = False
    test_set = build_dataset(dataset_name, cfg, split)

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    test_triplet(train_set, test_set, model, accuracy_calculator)


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            fewpascal/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # TODO: Add model info logging, i.e. params, memory, flops

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.TOKENS.ENABLE:
        num_prompts = len(cfg.TOKENS.PROMPTS)
    else:
        num_prompts = 1

    if cfg.TEST.FLIPPING:
        num_flips = 2
    else:
        num_flips = 1

    num_crops = cfg.TEST.NUM_SPATIAL_CROPS
    num_samples = num_prompts * num_flips * num_crops

    assert (
        test_loader.dataset.num_images
        % (num_prompts * num_flips * num_crops)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        test_loader.dataset.num_images // num_samples,
        num_samples,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    if cfg.MODEL.LOSS_FUNC in ["triplet_margin"]:
        perform_test_triplet(cfg, model)
    else:
        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    if writer is not None:
        writer.close()
