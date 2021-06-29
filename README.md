# Few Shot Classification

This repository is for training and evaluating few-shot and zero-shot learning approaches for image classification.

## Installation

Create a new conda environment for this repo.

```shell
$ conda create -n fewshot python=3.7`
```
Install pytorch 1.7 for your platform. If you are not using CUDA, or are using a different version on your machine, please refer to [these instructions](https://pytorch.org/get-started/previous-versions/).
```shell
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0`
```

Install the rest of the dependencies.

```shell
$ pip install -e .
```

Install the detectron2 package for visualization. Follow
[these instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Running a model

From the project directory, run

```shell
$ python3 tools/run_net.py --cfg configs/CLIP_zero.yaml
```

Run this for full options (e.g. distributed training)
```shell
$ python3 tools/run_net.py --help
```

Examine the `CLIP.yaml` file and the `fewpascal/configs/defaults.py` for further options.

