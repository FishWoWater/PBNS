#!/usr/bin/env python
# coding=utf-8

import os, time, os.path as osp
import numpy as np 
import open3d as o3d
import trimesh
from util import quads2tris
from IO import readPC2, readOBJ, writeOBJ
import argparse 
from fix_collision import fix_collisions
from pysdf import SDF
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--test_original', action='store_true')
parser.add_argument('--savedir', default='pbns_resize')
parser.add_argument('--fix_collision', action='store_true')
parser.add_argument('--test_posed', action='store_true')
args = parser.parse_args()

savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

if args.test_original:
    data = readPC2("./results/MyModel/body.pc2")
    cdata = readPC2("./results/MyModel/outfit.pc2")
elif args.test_posed:
    data = readPC2("./results/posed_model_female_range51/body.pc2")
    cdata = readPC2("./results/posed_model_female_range51/outfit.pc2")
else:
    data = readPC2("./results/MyModel2/body.pc2")
    cdata = readPC2("./results/MyModel2/outfit.pc2")

    # data = readPC2("./results/MyModel2_range5_2/body.pc2")
    # cdata = readPC2("./results/MyModel2_range5_2/outfit.pc2")

print(data.keys())
print(data['nSamples'], data['nPoints'])
print(cdata['nSamples'], cdata['nPoints'])

vertices = data['V']
cvertices = cdata['V']
print(vertices.shape, cvertices.shape)

if args.test_original:
    baseh_file = "./Model/tpose.obj"
    basec_file = "./Model/Outfit.obj"
elif args.test_posed:
    baseh_file = "./Model/tpose.obj"
    basec_file = "./Model/female_outfit.obj"
else:
    baseh_file = "./Model/tpose.obj"
    basec_file = "./Model/pbns_dense_tshirt.obj"
    basec_file = "./Model/Outfit2.obj"

# baseh = trimesh.load_mesh(baseh_file)
# basec = trimesh.load_mesh(basec_file)
# baseh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(baseh.vertices)), o3d.utility.Vector3iVector(np.asarray(baseh.faces)))
# basec = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(basec.vertices)), o3d.utility.Vector3iVector(np.asarray(basec.faces)))

baseh2v, baseh2f = readOBJ(baseh_file)
basec2v, basec2f = readOBJ(basec_file)
if args.test_original:
    baseh2f = quads2tris(baseh2f)
    basec2f = quads2tris(basec2f)
baseh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(baseh2v), o3d.utility.Vector3iVector(np.array(baseh2f)))
basec = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(basec2v), o3d.utility.Vector3iVector(np.array(basec2f)))

viser = o3d.visualization.Visualizer()

viser.create_window()
num_collisions = []

for fid in range(data['nSamples']):
    baseh.vertices = o3d.utility.Vector3dVector(vertices[fid])
    basec.vertices = o3d.utility.Vector3dVector(cvertices[fid])
    # o3d.visualization.draw_geometries([baseh, basec])
    if fid == 0:
        viser.add_geometry(baseh)
        viser.add_geometry(basec)
    else:
        viser.update_geometry(basec)
        viser.update_geometry(baseh)
        viser.poll_events()

    # o3d.io.write_triangle_mesh(str(fid) + '.obj', basec)
    # time.sleep(0.05)
    
    sdf = SDF(vertices[fid], np.asarray(baseh.triangles))
    is_contain = sdf.contains(cvertices[fid])
    num_collision = is_contain.sum()
    print('{} collisions detected'.format(num_collision))
    # save the mesh for rendering 
    savepath = osp.join(savedir, "%04d_garment.obj" % fid)
    rot = R.from_euler('x', [-90], degrees=True)
    cv = rot.apply(cvertices[fid]); hv = rot.apply(vertices[fid])
    if args.fix_collision:
        cv = fix_collisions(cv, hv, np.array(baseh2f))

    writeOBJ(savepath, cv, basec2f)
    # print('save to', savepath)
    savepath = osp.join(savedir, "%04d_body.obj" % fid)
    writeOBJ(savepath, hv, baseh2f)
    # print('save to', savepath)
    num_collisions.append(num_collision)

plt.hist(num_collisions)
plt.savefig(osp.basename(savedir) + '_collisions.png')
plt.show()

sns.distplot(num_collisions, kde=True)
plt.savefig(osp.basename(savedir) + '_collisionsns.png')
plt.show()
