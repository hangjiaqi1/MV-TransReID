# End-to-end Recovery of Human Shape and Pose

I modified code from https://github.com/akanazawa/hmr . I added 2D-to-3D color mapping for my human matching paper at [Here](https://arxiv.org/abs/2006.04569), and you are welcomed to check out them.  

The original paper is 

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik 'End-to-end Recovery of Human Shape and Pose' CVPR 2018

and my paper is

Zhedong Zheng, Nenggan Zheng and Yi Yang 'Parameter-Efficient Person Re-identification in the 3D Space' ArXiv 2021

[Project Page](https://akanazawa.github.io/hmr/)
![Teaser Image](https://github.com/layumi/hmr/blob/master/demo.png)

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
```
#### Install TensorFlow
With GPU:
```
conda install tensorflow-gpu==1.11.0
pip instal open3d 
```
Without GPU:
```
conda install tensorflow==1.11.0
pip instal open3d 
```

### Generate Market / Duke / MSMT
Please check the datapath before generation.
```bash
python generate_3DMarket_bg.py
python generate_3DDuke_bg.py
python generate_3DMSMT_bg.py
```

baseline without background
```bash
python generate_3DMarket.py
```

By defualt, I removed mesh information for fast data loading. 
If you want to preserve the mesh and visualize the 3D data, you could use the 
```bash
python demo_bg.py --img_path ../Market/pytorch/gallery/1026/1026_c1s6_038571_06.jpg
```
The output 3D data is `test.obj`. You could use `open3d` to visualize it. 
If you has one MacBook, you could visualize the `test.obj` in the folder. 
![](https://github.com/layumi/hmr/blob/master/hmr.png)

### Demo

1. Download the pre-trained models
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

2. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

Images should be tightly cropped, where the height of the person is roughly 150px.
On images that are not tightly cropped, you can run
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and supply
its output json (run it with `--write_json` option).
When json_path is specified, the demo will compute the right scale and bbox center to run HMR:
```
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
```
(The demo only runs on the most confident bounding box, see `src/util/openpose.py:get_bbox`)

### Training code/data
Please see the [doc/train.md](https://github.com/akanazawa/hmr/blob/master/doc/train.md)!

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```

### Opensource contributions
[Dawars](https://github.com/Dawars) has created a docker image for this project: https://hub.docker.com/r/dawars/hmr/

[MandyMo](https://github.com/MandyMo) has implemented a pytorch version of the repo: https://github.com/MandyMo/pytorch_HMR.git

[Dene33](https://github.com/Dene33) has made a .ipynb for Google Colab that takes video as input and returns .bvh animation!
https://github.com/Dene33/video_to_bvh 

<img alt="bvh" src="https://i.imgur.com/QxML83b.gif" /><img alt="" src="https://i.imgur.com/vfge7DS.gif" />
<img alt="bvh2" src=https://i.imgur.com/UvBM1gv.gif />

I have not tested them, but the contributions are super cool! Thank you!!


