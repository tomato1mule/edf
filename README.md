# Equivariant Descriptor Fields (EDFs)
PyTorch Implementation of Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning

## Installation

**Step 1.** We recommend using Anaconda, although not necessary.
```shell
git clone --recursive https://github.com/tomato1mule/edf.git
cd edf
conda create --name edf python=3.9.12 -y
conda activate edf
```
If using VSC, add python interpreter path: ~/anaconda3/envs/edf/bin/python

(ctrl+shift+P => Python: Select Interpreter => Enter interpreter path)


**Step 2.**
```shell
pip3 install -r requirements.txt
cd pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py                                           # You need g++ compiler installed in your system
cd ../../../..
conda develop ./pybullet-planning
conda develop .
```

**Step 3.**
```shell
pip3 install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip3 install torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip3 install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip3 install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip3 install e3nn==0.4.4 xitorch==0.3.0 iopath==0.1.9 fvcore==0.1.5.post20220504
pip3 install --no-index --no-cache-dir pytorch3d==0.7.0 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

## Generate Demo
```shell
PYTHONHASHSEED=0 python3 generate_demo.py --use-gui --file-name='mug_task_rim.gzip' --pick-type='rim'
# PYTHONHASHSEED=0 python3 generate_demo.py --use-gui --file-name='mug_task_mixed.gzip' --pick-type='mixed'
# PYTHONHASHSEED=0 python3 generate_demo.py --use-gui --file-name='mug_task_rim_lowvar.gzip' --pick-type='rim' --low-var
# PYTHONHASHSEED=0 python3 generate_demo.py --use-gui --file-name='mug_task_rim_5_demo.gzip' --pick-type='rim' --seeds 0 1 2 3 4
# PYTHONHASHSEED=0 python3 generate_demo_stick.py --use-gui --file-name='stick_task_rim.gzip' --pick-type='rim'
```

## Train
```shell
bash experiments/train.sh                  # EDF 10 demo
# bash experiments/train_5_demo.sh         # EDF 5 demo
# bash experiments/train_ablation.sh       # IDF 10 demo
# bash experiments/train_lowvar.sh         # EDF lowvar 10 demo
# bash experiments/train_multimodal.sh     # EDF mixed mug task 10 demo
# bash experiments/train_stick.sh          # EDF stick_task 10 demo
# bash experiments/baseline_train.sh       # SE(3) Transporter Networks 10 demo
```

## Eval
```shell
bash experiments/eval.sh                          # EDF 
# bash experiments/eval_ablation.sh                # IDF
# bash experiments/eval_multimodal.sh              # EDF mixed mug task
# bash experiments/eval_stick.sh                   # EDF stick task 
```



## Reproducibility
Our algorithms are guaranteed to be fully deterministic in a local machine.
However, the results may be different accross different machines.
This is presumably due to the numerical differences between processors.
Unfortunately, even a very tiny numerical difference would result in completely different output due to the MCMC steps.
Therefore, we provide download links to the task demonstrations and checkpoints.
Please download and use these files for reproducing our results.

### Download links:

Demos: https://drive.google.com/file/d/1xtP1RFgXqvlK9_K6t0N_cw4cY3Ydlqlp/view?usp=sharing

Checkpoints: https://drive.google.com/file/d/1m5ytgbfSDjO-muxgceiEm2Y9pUyq95Gr/view?usp=sharing

Checkpoints for baseline method (SE(3) Transporter Networks): https://drive.google.com/file/d/1wJ3O3Wo2yqeLUmnQOWEOepwzvRHSD70U/view?usp=sharing

Please unzip these files in current directory.

Next, move checkpoint/\<demo_name_to_experiment\>/train_pick to checkpoint/train_pick and checkpoint/\<demo_name_to_experiment\>/train_place to checkpoint/train_place.



## Logs
We also provide the training and evalutation logs.

### Download links:
Logs: https://drive.google.com/file/d/1OqiJhI_Pn_ICZREH_28e8_XYRF4o9oIf/view?usp=sharing

Logs for baseline method (SE(3) Transporter Networks): https://drive.google.com/file/d/1qyyupWYU0UpW4a616O0bKoAsA0VPJS6Z/view?usp=sharing





