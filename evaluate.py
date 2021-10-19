from data_processing.evaluation import eval_mesh, eval_meshOnet
import trimesh
import pickle as pkl
import os

from multiprocessing import Pool
import argparse
from glob import glob
import traceback
import random
import numpy as np


def eval(path):
    eval_file_name = "/eval.pkl"
    try:
        if os.path.exists(path + eval_file_name):
            print('File exists. Done.')
            return

        path = os.path.normpath(path)
        folder = path.split(os.sep)[-2]
        file_name = path.split(os.sep)[-1]


        pred_mesh_path = path + '/surface_reconstruction.off'
        pred_mesh = trimesh.load(pred_mesh_path, process=False)

        if args.data_type == 'ifnet':
            gt_mesh_path = data_path + '/{}/{}/isosurf_scaled.off'.format(folder, file_name)
        elif args.data_type =='human':
            gt_mesh_path = data_path + '/{}.ply'.format(file_name)
        else:
            raise ValueError('Unknown data type: ' + args.data_type)
        gt_mesh = trimesh.load(gt_mesh_path, process=False)

        if args.data_type == 'human':
            input_points = gt_mesh.vertices  # pv.read(path).points

            vmax = np.max(input_points)
            vmin = np.min(input_points)

            bbox_length = vmax - vmin
            bbox_middle = (vmax + vmin) / 2
            input_points -= bbox_middle
            input_points /= bbox_length
            gt_mesh.vertices = input_points

        eval = eval_mesh(pred_mesh, gt_mesh, min, max)


        pkl.dump(eval ,open(path + eval_file_name, 'wb'))
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


def evalOnet(path):
    eval_file_name = "/evalOnet.pkl"
    try:
        if os.path.exists(path + eval_file_name):
            print('File exists. Done.')
            return
        else:
            path = os.path.normpath(path)
            folder = path.split(os.sep)[-2]
            file_name = path.split(os.sep)[-1]

            pred_mesh_path = path + '/surface_reconstruction.off'
            pred_mesh = trimesh.load(pred_mesh_path, process=False)

            gt_path = data_path + '/{}/{}'.format(folder, file_name)

            input_path = gt_path + '/pointcloud.npz'
            input_pointcloud_dict = np.load(input_path)
            surface_points = input_pointcloud_dict['points'].astype(np.float32)
            surface_normals = input_pointcloud_dict['normals'].astype(np.float32)

            supervision_path = gt_path + '/points.npz'
            sup_points_dict = np.load(supervision_path)
            sup_points = sup_points_dict['points']
            sup_points = sup_points.astype(np.float32)
            occupancies = sup_points_dict['occupancies']
            occupancies = np.unpackbits(occupancies)[:sup_points.shape[0]]

            gt_dict = {'surface_points': surface_points, 'surface_normals': surface_normals,
                       'iou_points': sup_points, 'iou_occ': occupancies}
            eval_res = eval_meshOnet(pred_mesh, gt_dict)

            pkl.dump(eval_res ,open(path + eval_file_name, 'wb'))
            print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run input evaluation'
    )

    parser.add_argument('-generation_path', type=str)
    parser.add_argument('-dataset_type', required=True)


    args = parser.parse_args()

    if args.data_type == 'ifnet':
        data_path = 'shapenet/data'
    elif args.data_type == 'onet':
        data_path = 'occupancy_networks/data/ShapeNet'
    elif args.data_type == 'human':
        data_path = 'MPI-FAUST/test/scans' #TODO fix path
    else:
        raise ValueError('Unexpected data type: ' + str(args.data_type))


    min = -0.5
    max = 0.5

    p = Pool(10)
    if args.data_type == 'human':
        paths = glob(args.generation_path + '/scans/*/')
    else:
        paths = glob(args.generation_path + '/*/*/')


    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(paths)
    if args.data_type == 'ifnet' or args.data_type == 'human':
        p.map(eval, paths)
    else:
        p.map(evalOnet, paths)
    p.close()
    p.join()
