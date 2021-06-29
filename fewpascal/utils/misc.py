#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import math
from datetime import datetime
import torch
from iopath.common.file_io import g_pathmgr

import fewpascal.utils.logging as logging
import fewpascal.utils.multiprocessing as mpu

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            fewpascal/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            fewpascal/config/defaults.py
        cur_epoch (int): current epoch.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    else:
        return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def get_class_names(path):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
    Returns:
        class_names (list of strs): list of class names.
    """
    try:
        with g_pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    return class_names
