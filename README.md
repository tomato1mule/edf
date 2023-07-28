# [ICLR 2023] Equivariant Descriptor Fields (EDFs)

Official PyTorch implementation of Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning (ICLR 2023 Poster).

The paper can be found at: https://openreview.net/forum?id=dnjZSPGmY5O

> **Note**\
> This is a standalone implementation of EDFs without PyBullet simulation environments. To reproduce our experimental results in the paper, please check the following branch:  https://github.com/tomato1mule/edf/tree/iclr2023_rebuttal_ver

> **Note**\
> EDF+ROS MoveIt example (unstable): https://github.com/tomato1mule/edf_pybullet_ros_experiment


## Installation

**Step 1.** Clone Github repository.
```shell
git clone https://github.com/tomato1mule/edf
```

**Step 2.** Setup Conda environment.
```shell
conda create -n edf python=3.8
conda activate edf
```

**Step 3.** Install Dependencies
```shell
CUDA=cu113
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install iopath fvcore
pip install --no-index --no-cache-dir pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_${CUDA}_pyt1110/download.html
pip install -e .
```

# Usage
## Train
```shell
python pick_train.py
python place_train.py
```

If you want to load already trained checkpoints, please rename 'checkpoint_example' folder to 'checkpoint'.
## Evaluate
Please run the example notebook codes for visualizing sampled poses from trained models (evaluate_pick.ipynb and evaluate_place.ipynb)

## View train log
```shell
python train_log_viewer.py --logdir="checkpoint/mug_10_demo/ {pick or place} /trainlog_iter_{iter}.gzip"
```

# Citing
If you find our paper useful, please consider citing our paper:
```
@inproceedings{
ryu2023equivariant,
title={Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning},
author={Hyunwoo Ryu and Hong-in Lee and Jeong-Hoon Lee and Jongeun Choi},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=dnjZSPGmY5O}
}
```



