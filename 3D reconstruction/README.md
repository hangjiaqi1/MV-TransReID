# 3D reconstruction

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3, demo alone runs with TF 1.12

### Installation

#### Setup virtualenv
```
conda create --name hmr python=2.7
conda activate hmr
pip install numpy
pip install -r requirements.txt
conda install tensorflow-gpu==1.11.0
pip instal open3d 
```


### Demo

1. Download the pre-trained models

   Download URL: [Google disk](https://drive.google.com/file/d/19nymcyBRBL5i0i-TByR9O4yTeuNU5DA5/view?usp=sharing)
   
   unzip under path: ./MV-TransReID/3D reconstruction.

3. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```


