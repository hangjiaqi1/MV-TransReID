"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import os

from PIL import Image
from src.util import renderer as vis_util
from src.util import image as img_util
from src.tf_smpl import projection as proj_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', '../MSMT/pytorch/train/0000/0000_032_09_0303morning_0029_0.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        #vert_shifted, cam=None, img=img, do_alpha=True)
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # plt.ion()
    fig = plt.figure()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    result = Image.fromarray(rend_img)
    result.save('mesh.jpg')
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    fig.savefig('demo.jpg')
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None, fliplr=False):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if fliplr:
        img = np.fliplr(img)

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    for fliplr in [False]:
        input_img, proc_param, img = preprocess_image(img_path, json_path, fliplr=fliplr)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
       # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        print("img.shape",img.shape)
        print("input_img.shape",input_img.shape)
        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)
        print("img.shape",img.shape)
        print("input_img.shape",input_img.shape)


    # scaling and translation
    print("vert.shape.....",verts[0][:, :2].shape)
    save_mesh(img, img_path, proc_param, joints[0], verts[0], cams[0])
    visualize(img, proc_param, joints[0], verts[0], cams[0])

def save_mesh(img, img_path, proc_param, joints, verts, cam):
    cam_for_render, vert_3d, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    cam_for_render, vert_shifted = cam, verts
    #print(proc_param)
    #print(vert_shifted)
    camera  = np.reshape(cam_for_render, [1,3])
    print("camera",camera)
    w, h, _ = img.shape
    print("img.shape",img.shape)
    imgsize = max(w,h)
    # project to 2D
    print("vert_2d.shape",verts[:, :2].shape)
    vert_2d = verts[:, :2] + camera[:, 1:]
    print("vert_2d.shape",vert_2d.shape)
    vert_2d = vert_2d * camera[0,0]
    print("vert_2d.shape",vert_2d.shape)
    img_copy = img.copy()
    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = 'test.obj'
    foreground_index_2d = np.zeros((w,h))+99999
    foreground_value_2d = np.zeros((w,h))+99999
    with open(obj_mesh_name, 'w') as fp:
        # Decide Forground
        print("vert_2d.shape",vert_2d.shape)
        for i in range(vert_2d.shape[0]):
            v2 = vert_2d[i,:]
            v3 = vert_3d[i,:]
            z = v3[2]
            x = int(round( (v2[1]+1)*0.5*imgsize ))
            y = int(round( (v2[0]+1)*0.5*imgsize ))
            if w<h:
                x = int(round(x -h/2 + w/2))
            else:
                y = int(round(y - w/2 + h/2))
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            if z < foreground_value_2d[x,y]:
                foreground_index_2d[x,y] = i
                foreground_value_2d[x,y] = z
        # Draw Color
        for i in range(vert_2d.shape[0]):
            v2 = vert_2d[i,:]
            v3 = vert_3d[i,:]
            z = v3[2]
            x = int(round( (v2[1]+1)*0.5*imgsize ))
            y = int(round( (v2[0]+1)*0.5*imgsize ))
            if w<h:
                x = int(round(x -h/2 + w/2))
            else:
                y = int(round(y - w/2 + h/2))
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            if i == foreground_index_2d[x,y]: 
                c = img[x, y, :]/255.0
                img_copy[x,y,:] = 0
            else:
                c = [1,1,1] 
            fp.write( 'v %f %f %f %f %f %f\n' % ( v3[0], v3[1], v3[2], c[0], c[1], c[2]) )
        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
    img_copy = Image.fromarray(img_copy, 'RGB')
    img_copy.save('input.png')

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
