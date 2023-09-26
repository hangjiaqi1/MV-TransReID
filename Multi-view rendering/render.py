#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
import json
'''
if torch.__version__ == '1.6.0+cu101' and sys.platform.startswith('linux'):
    get_ipython().system('pip install pytorch3d')
else:
    need_pytorch3d = False
    try:
        import pytorch3d
    except ModuleNotFoundError:
        need_pytorch3d = True
    if need_pytorch3d:
        get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
        get_ipython().system('tar xzf 1.10.0.tar.gz')
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")
'''
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)
import matplotlib.pyplot as plt
import matplotlib
from utils import Params


# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Import parameters
params = Params("params.json")
obj_filename = params.obj_filename

#azim tensor([[-270., -240., -210., -180., -150., -120.,  -90.,  -60.,  -30.,    0., 30.,   60.]])
#dist tensor([[2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000]])
#elev tensor([[35., 35., 35., 35., 35., 35., 35., 35., 35., 35., 35., 35.]])

def get_mesh(obj_filename, device):
    """
    Generates Meshes object and initializes the mesh with vertices, faces,
    and textures.

    Args:
        obj_filename: str, path to the 3D obj filename
        device: str, the torch device containing a device type ('cpu' or
        'cuda')

    Returns:
        mesh: Meshes object
    """
    # Get vertices, faces, and auxiliary information
    verts, faces, aux = load_obj(
        obj_filename,
        device=device,
        load_textures=False,
        create_texture_atlas=False,
        # texture_atlas_size=4,
        # texture_wrap="repeat"
         )
    # Create a textures object
    # atlas = aux.texture_atlas
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    # 将顶点坐标减去均值，实现中心化处理。keepdim=True表示保持维度不变。
    verts -= verts.mean(0, keepdim=True)
    # 将顶点坐标除以最大值，实现归一化处理。
    verts /= verts.max()
    # 将物体倒过来
    verts = -verts
    # Create Meshes object
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=textures) 
    return mesh


def get_renderer(image_size, dist, device, elev, azim):
    """
    Generates a mesh renderer by combining a rasterizer and a shader.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    # Initialize the camera with camera distance, elevation, azimuth angle,
    # and image size
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # Initialize rasterizer by using a MeshRasterizer class
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    #在物体前面放置点光源
    lights = PointLights(
        device=device,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0.0, 0.0, 1e5),) #这里将光摆在物体面前，就会形成高光
    )
    # The textured phong shader interpolates the texture uv coordinates for
    # each vertex, and samples from a texture image.
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    # Create a mesh renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(rasterizer, shader)
    return renderer


def render_image(renderer, mesh, obj_filename, azim, elev):
    """
    Renders an image using MeshRenderer class and Meshes object. Saves the
    rendered image as a .png file.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    image = renderer(mesh)
    # 像素归一化
    image = torch.clamp(image, 0.0, 1.0)
    dir_to_save = "out"
    os.makedirs(dir_to_save, exist_ok=True)
    out = os.path.normpath(obj_filename).split(os.path.sep)
    mesh_filename = out[-1].split(".")[0]
    sep = '_'
    file_to_save = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(mesh_filename, sep,
                                                     "elev", int(elev),
                                                     sep, "azim",
                                                     int(azim), ".png")
    filename = os.path.join(dir_to_save, file_to_save)
    matplotlib.image.imsave(filename, image[0, ..., :3].cpu().numpy())
    print("Saved image as " + str(filename))
    

def compile_all_steps(image_size, dist, device, elev, azim, obj_filename):
    """
    Combines the above steps.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values
        obj_filename: str, path to the 3D obj filename

    Returns:
        None
    """
    renderer = get_renderer(image_size, dist, device, elev, azim)
    mesh = get_mesh(obj_filename, device)
    render_image(renderer, mesh, obj_filename, azim, elev)
    return None

'''
def main():
     [compile_all_steps(params.image_size, params.camera_dist, device, x, y,
                        params.obj_filename)
         for x in params.elevation for y in params.azim_angle]
'''
def main():
    [compile_all_steps(params.image_size, params.camera_dist, device, elevation, azim_angle, params.obj_filename)
     for elevation, azim_angle in zip(params.elevation, params.azim_angle)]


if __name__ == "__main__":
    main()
