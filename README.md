# [ICLR 2023] Equivariant Descriptor Fields (EDFs)

Official PyTorch implementation of Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning (ICLR 2023 Poster).

The paper can be found at: https://openreview.net/forum?id=dnjZSPGmY5O

## Installation

**Step 1.** We recommend using Anaconda, although not necessary.
```shell
git clone --recursive anon
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
python setup.py
cd ../../../..
conda develop ./pybullet-planning
conda develop .
```

**Step 3.**
```shell
pip3 install -e . --extra-index-url https://download.pytorch.org/whl/cu113 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```


## Generate Demo
```shell
python3 python3 generate_demo.py
```

## Train
```shell
bash experiments/train.sh           # Train mug-hanging task with rim grasp, 10 demo.
# bash experiments/train_bowl.sh    # Train bowl pick-and-place task, 10 demo.
# bash experiments/train_bottle.sh  # Train bottle pick-and-place task, 10 demo.

# bash experiments/train_1_demo.sh      # Train mug-hanging task with rim grasp, 1 demo.
# bash experiments/train_5_demo.sh      # Train mug-hanging task with rim grasp, 5 demo.
# bash experiments/train_lowvar.sh      # Mug-hanging task, 10 demo, but with low-variance, unimodal demos.
# bash experiments/train_multimodal.sh  # Mug-hanging task, 10 demo, but with high-variance, multimodal demons.

# bash experiments/train_ablation.sh    # Mug-hanging task with NDF-like type-0 only model, 10 demo.
# bash experiments/train_ablation.sh    # Bowl pick-and-place task with NDF-like type-0 only model, 10 demo.
# bash experiments/train_ablation.sh    # Bottle pick-and-place task with NDF-like type-0 only model, 10 demo.
# bash experiments/baseline_train.sh    # Mug-hanging task with SE(3) Transporter Networks, 10 demo.
# bash experiments/baseline_train_bowl.sh    # Bowl pick-and-place task with SE(3) Transporter Networks, 10 demo.
# bash experiments/baseline_train_bottle.sh    # Bottle pick-and-place task with SE(3) Transporter Networks, 10 demo.

```

## Eval
```shell
bash experiments/eval_gui.sh     # This will run evaluation for mug-task.
                                 # Modify experiments/eval_gui.sh to evaluate for bowl- and bottle-tasks.
                                 # Please don't forget to change the checkpoint files.
                                 # They should be located in checkpoint/train_pick and checkpoint/train_place  


```
In the experiments folder, one can find many other evaluation scripts that were used in the paper.



## Reproducibility
Our implementation is fully deterministic in a local machine when the seeds are properly fixed.
However, the results may be different accross different machines.
This is due to the tiny numerical differences between different processors.
Unfortunately, even a very small numerical difference would result in completely different output due to the MCMC steps.
Therefore, we provide download links to the task demonstrations and checkpoints.
Please download and use these files if reproducbility matters.

Checkpoint: https://drive.google.com/file/d/1TYJ0aJKe1bNPv8QZ6-4Kn2L6K-L246qV/view?usp=share_link

Demo: https://drive.google.com/file/d/1bTS6goVQw_ihEqDumox09tpc6daE6Gu8/view?usp=sharing

please unzip these files and place the contents in 'checkpoint/' and 'demo/' directory, individually.







