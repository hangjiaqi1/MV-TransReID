import numpy as np
import open3d as o3d

mesh= o3d.io.read_triangle_mesh('test.obj', True)
print(">>>>>>>>>>>>>>>>>>>>>>",mesh)
obj = np.asarray(mesh.vertices, dtype=np.float32)
obj -= np.mean(obj, axis=0)
print("obj.shape",obj.shape)
# obj_color = np.asarray(mesh.vertex_colors, dtype=np.float32)
# print("obj_color.shape",obj_color.shape)
# obj = np.concatenate((obj, obj_color), axis=1)
print("obj.shape",obj.shape)


o3d.visualization.draw_geometries([mesh])