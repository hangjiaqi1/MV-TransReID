# MV-TransReID Multi-view rendering
## 1.install Pytorch3d
The core library is written in PyTorch. Several components have underlying implementation in CUDA for improved performance. A subset of these components have CPU implementations in C++/PyTorch. It is advised to use PyTorch3D with GPU support in order to use all the features.
- Linux or macOS or Windows
- Python 3.8, 3.9 or 3.10
- PyTorch 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 2.0.0 or 2.0.1.
- torchvision that matches the PyTorch installation. You can install them together as explained at pytorch.org to make sure of this.
- gcc & g++ ≥ 4.9
- fvcore
- ioPath
- If CUDA is to be used, use a version which is supported by the corresponding pytorch version and at least version 9.2.
- If CUDA older than 11.7 is to be used and you are building from source, the CUB library must be available. We recommend version 1.10.0.
The runtime dependencies can be installed by running:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For the CUB build time dependency, which you only need if you have CUDA older than 11.7, if you are using conda, you can continue with
```
conda install -c bottler nvidiacub
```
Otherwise download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice. Define the environment variable CUB_HOME before building and point it to the directory that contains CMakeLists.txt for CUB. For example on Linux/Mac,
```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```

## 2.Multi-view rendering
```
python render.py
```
