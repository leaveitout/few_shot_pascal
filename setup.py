#!/usr/bin/env python3
"""Setup the fewpascal package"""
from setuptools import find_packages, setup

setup(
    name="fewpascal",
    version="0.0.0",
    author="Seán Bruton",
    author_email="seanbruton2011@gmail.com",
    url="https://github.com/leaveitout/few_shot_pascal",
    description="Few-shot, zero-shot learning on Pascal classes",
    long_description="Few-shot, zero-shot learning on Pascal classes",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "fvcore",
        "numpy",
        "iopath",
        "torch",
        "simplejson",
        "tqdm",
        "opencv-python-headless",
        "torchvision",
        "clip",
    ],
    packages=find_packages(exclude=("configs", "tests", "tools")),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
