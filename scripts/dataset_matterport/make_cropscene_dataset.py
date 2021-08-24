""" load ply files and output the dataset accepted by ConvONet
        (only for running generate_optim_largescene.py)
"""

import os
import argparse
import numpy as np
from glob import glob
from plyfile import PlyData


def read_point_ply(filename):
    pd = PlyData.read(filename)['vertex']
    v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
    try:
        n = np.array(np.stack([pd[i] for i in ['nx', 'ny', 'nz']], axis=-1))
    except:
        print(f"warning: cannot find normals in file {filename}")
        n = np.ones_like(v)
        n = n / (np.linalg.norm(n, axis=1).reshape([-1, 1]) + 1e-6)
    return v, n


def make_temp_dataset(in_folder, out_folder, do_norm=True):

    os.makedirs(out_folder, exist_ok=True)

    all_pc_list = glob(os.path.join(in_folder, "*.ply"))
    for pc_path in all_pc_list:
        
        points, normals = read_point_ply(pc_path)
        vert_max = points.max(axis=0)
        vert_min = points.min(axis=0)
        if do_norm:
            loc = (vert_min + vert_max) / 2
            scale = (vert_max - vert_min).max()
        else:
            loc = np.array([0, 0, 0], dtype=np.float64)
            scale = np.array([1.0], dtype=np.float64)
        print('loc', loc, 'scale', scale)

        points = (points - loc) / scale
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        obj_name = os.path.basename(pc_path)[:-4] #modelname.ply
        os.makedirs(os.path.join(out_folder, "large_scenes", obj_name), exist_ok=True)

        save_npz_path = os.path.join(out_folder, "large_scenes", obj_name, "pointcloud.npz")
        npz_dict = dict(points=points.astype(np.float16),
                        normals=normals.astype(np.float16),
                        loc=loc.astype(np.float64),
                        scale=scale.astype(np.float64))
        np.savez(save_npz_path, **npz_dict)
        print(f"we save pointcloud npz to: {save_npz_path}")

        save_npz_path = os.path.join(out_folder, "large_scenes", obj_name, "points_iou.npz")
        #if not do_norm: raise
        NUM_SPATIAL_PTS = 1000000 # 100w
        RATIO_SUR = 0.25 
        RATIO_STD = 0.75
        if do_norm:
            STD = 0.1 
        else:
            STD = 0.1 * ((vert_max - vert_min).max())
        spatial_points_xyz = np.concatenate([points[np.random.choice(len(points), size=(int(NUM_SPATIAL_PTS * RATIO_STD), ))] + \
                                                np.random.randn(int(NUM_SPATIAL_PTS * RATIO_STD), 3) * STD,
                                             np.random.rand(int(NUM_SPATIAL_PTS * (1-RATIO_SUR-RATIO_STD)), 3)*1.1-0.55, 
                                             points[np.random.choice(len(points), size=(int(NUM_SPATIAL_PTS*RATIO_SUR), ))]], axis=0)
        spatial_points_occ = np.concatenate([np.ones([int(NUM_SPATIAL_PTS * (1-RATIO_SUR))]), 
                                                      np.zeros([int(NUM_SPATIAL_PTS*RATIO_SUR)])], axis=0) # # dummy (0 for surface, 1 for space)
        shuffle_index = np.random.permutation(NUM_SPATIAL_PTS)
        spatial_points_xyz = spatial_points_xyz[shuffle_index]
        spatial_points_occ = spatial_points_occ[shuffle_index]
        npz_dict = dict(points=spatial_points_xyz.astype(dtype=np.float16), # for normalization
                        occupancies=np.packbits(spatial_points_occ.astype(dtype=bool)), 
                        z_scale=np.array(0).astype(np.float64),
                        semantics=np.zeros([NUM_SPATIAL_PTS], dtype=np.int64)) 
        np.savez(save_npz_path, **npz_dict) # dummy file
        print(f"we save points_iou npz to: {save_npz_path}")

    test_lst_path = os.path.join(out_folder, "large_scenes", "test.lst")
    with open(test_lst_path, "w") as f:
        [f.write(os.path.basename(s)[:-4] + "\n") for s in all_pc_list]
    print(f"we save test list to: {test_lst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_folder', type=str, default="./my_dataset", help="the folder path containing point cloud ply files")
    parser.add_argument('--out_folder', type=str, default="./my_dataset_rec", help="the output dir")
    parser.add_argument('--do_norm', action="store_true", help="do normalization within -0.5 ~ +0.5")
    args = parser.parse_args()

    make_temp_dataset(args.in_folder, args.out_folder, do_norm=args.do_norm)