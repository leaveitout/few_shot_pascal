#!/usr/bin/env python3

"""Wrapper to train and test a video classification model."""
from fewpascal.config.defaults import assert_and_infer_cfg
from fewpascal.utils.misc import launch_job
from fewpascal.utils.parser import load_config, parse_args

from test_net import test
from train_net import train
# TODO: Enable when complete
# from visualization import visualize
# from demo_net import demo


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # TODO: Enable when completed.
    # Perform model visualization.
    # if cfg.TENSORBOARD.ENABLE and (
    #     cfg.TENSORBOARD.MODEL_VIS.ENABLE or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    # ):
    #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # TODO: Enable when completed.
    # Run demo.
    # if cfg.DEMO.ENABLE:
    #     demo(cfg)


if __name__ == "__main__":
    main()
