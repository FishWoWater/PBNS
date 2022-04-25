#!/usr/bin/env python
# coding=utf-8

import trimesh 
import joblib
import numpy as np
from pysdf import SDF
from fix_collision import fix_collisions
 # cpos = fix_collisions(cpos, hpos, hfaces)

num_example_faces = 5676; num_total_faces = 10498
num_example_vertices = 2904; num_total_vertices = 5317

cv_indices = list(range(num_example_vertices)); hv_indices = list(range(num_example_vertices, num_total_vertices))
cf_indices = list(range(num_example_faces)); hf_indices = list(range(num_example_faces, num_total_faces))

def eval_rollout_collision(rollout_path, key='pred_pos', last_frame_only=False):
    print('evaluating collision of rollout path', rollout_path)
    if last_frame_only: print('only considering last frame')
    trajs = joblib.load(rollout_path)
    num_trajs = len(trajs)

    num_collisions = []
    for trajid in range(num_trajs):
        traj = trajs[trajid]

        wpos = traj[key][0]
        faces = traj['faces'][0]
        trajlen_by_faces = len(traj['faces']) - 1
        cpos = wpos[cv_indices]; hpos = wpos[hv_indices]
        cfaces = faces[cf_indices]; hfaces = faces[hf_indices] - num_example_vertices

        sdf = SDF(hpos, hfaces)
        num_collision = 0

        for fid in range(len(traj[key])):
            if last_frame_only and fid != len(traj[key]) - 1:
            # if last_frame_only and fid != 50:
                continue
            wpos = traj[key][fid]
            faces = traj['faces'][min(trajlen_by_faces, fid)]
            cpos = wpos[cv_indices]
            # cpos = fix_collisions(cpos, hpos, hfaces)
            is_contained = np.array(sdf.contains(cpos))
            num_collision += is_contained.sum()

        num_collisions.append(num_collision)

    print('collisions for all trajectories', num_collisions)
    num_collisions = np.array(num_collisions)
    print('avg/min/max collision is', num_collisions.mean(), num_collisions.min(), num_collisions.max())
    return num_collisions.mean(), num_collisions

if __name__ == '__main__':
    import sys 
    inpath = sys.argv[1] if len(sys.argv) > 1 else "../checkpoints/rw02.pkl"
    # eval_rollout_collision(inpath)

    import glob
    import os, os.path as osp 
    inpaths = glob.glob("../checkpoints/rollout*lr5em5*.pkl")
    # for inpath in inpaths:
    for inpath in [inpath]:
        print('evaluating coliison for', inpath)
        eval_rollout_collision(inpath, key='pred_pos', last_frame_only=True)
        cmd = 'feh {}'.format(inpath.replace(".pkl", '.png'))
        # os.system(cmd)

