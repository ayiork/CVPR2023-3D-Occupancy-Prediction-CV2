# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**0. LISA**
```shell
ssh -X lcurXXXX@lisa.surfsara.nl
module purge
module load 2021
module load Anaconda3/2021.05
```
**1. LISA**
open the .bashrc file (it should be in your home directory) and add the following lines to the end of the file and save it:\
module load 2022; \
module load CUDA/11.6.0 \
Run the following command to reserve a GPU node:\
```shell
srun -u --pty --gpus=1 --mem=20G --cpus-per-task=4 --time=1:00:00 -p gpu_titanrtx_shared_course bash -i
```

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n cv2 python=3.8 -y
source activate cv2
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
# pip install mmcv-full==1.4.0
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**d2. Install some specific versions for some libraries**\
Not sure if this is necessary. (suggested from the TA but I changed the version of numpy)
```shell
pip install scikit-image==0.17.2
pip install numpy==1.19.5
pip install matplotlib==3.5.2
pip install pandas==1.2.5
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install timm.**
```shell
pip install timm
```


**g. Clone BEVFormer.**
```
git clone https://github.com/fundamentalvision/BEVFormer.git
```

**h. Prepare pretrained models.**
```shell
cd BEVFormer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

note: this pretrained model is the same model used in [detr3d](https://github.com/WangYueFt/detr3d)