# Equivariant Descriptor Fields (EDFs)
PyTorch Implementation for the EDFs.
The paper can be found at: https://arxiv.org/abs/2206.08321

## Installation

**Step 1.**
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
python setup.py
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
pip3 install --no-index --no-cache-dir pytorch3d==0.6.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

## Reproducibility
For the reproducibility, we fixed all the seeds for the random number generators and used deterministic algorithms only.
We also provide pickles of the tensor product layers as there is some numerical nondeterminism (of order 1e-6) in the initialization of E3NN that cannot be controlled by simply setting the seeds.

Since our algorithm heavily relies on MCMC, very small errors may accumulate to result in huge differences.
Unfortunately, there are small numerical differences across different platforms for some modules.
As a result, the reproducibility of the algorithm is not guaranteed across different platforms.
Nevertheless, reproducibility is at least guaranteed in the same platform.
Therefore, we provide the checkpoints for the trained models. 
```shell
mkdir checkpoint
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GyIz-u928OLP9myUC31QV3rHayyz48J3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GyIz-u928OLP9myUC31QV3rHayyz48J3" -O train_pick_reproducible.zip && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_54umYhNJTwEThPPgQUnah9zjog_ap7C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_54umYhNJTwEThPPgQUnah9zjog_ap7C" -O train_place_reproducible.zip && rm -rf ~/cookies.txt
unzip train_pick_reproducible.zip
unzip train_place_reproducible.zip
cd ..
```
We also provide the train/test datasets.
```shell
mkdir demo
cd demo
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HhusCuLTSYrm4b1mh0MN9nd8s3C5ZsgV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HhusCuLTSYrm4b1mh0MN9nd8s3C5ZsgV" -O mug_task.zip && rm -rf ~/cookies.txt
unzip mug_task.zip 
cd ..
```





