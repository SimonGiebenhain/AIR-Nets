import trimesh
import numpy as np
#import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

ROOT = 'shapenet/data'


def boundary_sampling(path):
    try:

        if os.path.exists(path + '/surface_{}_samples.npy'.format(n_samples)):
            return

        off_path = path + '/isosurf_scaled.off'

        out_file_surface = path + '/surface_{}_samples.npy'.format(n_samples)

        mesh = trimesh.load(off_path)
        points_surface = mesh.sample(n_samples)

        np.save(out_file_surface, points_surface)

        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':

    n_samples = 30000

    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( ROOT + '/*/*'))