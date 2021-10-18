from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree as KDTree

import random


def create_grid_points_from_bounds(minimun, maximum, res, scale=None):
    if scale is not None:
        res = int(scale * res)
        minimun = scale * minimun
        maximum = scale * maximum
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


class ShapeReconstructionDataset(Dataset):
    """
    Datasets that loads point clouds sampled from the surface of an object.

    Attributes:
        mode str: either 'train', 'val' or 'test', determines from which split to sample data from
        nsamples_input int: number of points in input point cloud
        nsamples_supervision int: number of supervision points i.e. points for which loss is calculated
        single_view bool: when False, points are sampled uniformly on the surface, when True, object is observed from a
            single view point from the upper hemisphere
        test_equiv bool: False when training, when set to True input is randomly rotated in order to investigate
            rotation equivariance of model.
    """

    def __init__(self, mode, dataset_type, data_path, nsamples_input = 300, split_file ='shapenet/split.npz',
                 batch_size = 64, nsamples_supervision = 1024, num_workers = 10, sample_distribution = [0.5, 0.5], sample_sigmas = [0.1, 0.01], noise_std=None,
                 voxelize=False):

        assert dataset_type in ['onet', 'ifnet']
        self.dataset_type = dataset_type

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.mode = mode
        self.data = self.split[mode]

        self.nsamples_supervision = nsamples_supervision
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nsamples_input = nsamples_input

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * nsamples_supervision).astype(np.uint32)

        self.noise_std = noise_std
        self.voxelize = voxelize
        if voxelize:
            self.res = 128
            self.grid_points = create_grid_points_from_bounds(-0.5, 0.5, self.res)
            self.kdtree = KDTree(self.grid_points)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]
        if self.dataset_type == 'ifnet':
            pc_path = path + '/surface_{}_samples.npy'.format(30000)
            obs = np.load(pc_path)
        else:
            pc_path = path + '/pointcloud.npz'
            input_pointcloud_dict = np.load(pc_path)
            obs = input_pointcloud_dict['points'].astype(np.float32)

        # sample observations
        input_idx = np.random.randint(0, obs.shape[0], self.nsamples_input)
        obs = obs[input_idx]

        # sample supervision coordinates and corresponding ground truth occupancies
        if self.dataset_type == 'ifnet':
            sup_occupancies = []
            sup_coords_vox = []
            sup_coords = []
            for i, num in enumerate(self.num_samples):
                boundary_samples_path = path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_occupancies = boundary_samples_npz['occupancies']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                #if self.voxelize:
                #    #TODO is flipping axes necessary conform with onet data or unnecessary?
                #    boundary_sample_coords = boundary_sample_points.copy()
                #    boundary_sample_coords[:, 0], boundary_sample_coords[:, 2] = boundary_sample_points[:, 2], boundary_sample_points[:, 0]
                #    boundary_sample_coords = 2 * boundary_sample_coords
                #    sup_coords_vox.extend(boundary_sample_coords[subsample_indices])
                sup_coords.extend(boundary_sample_points[subsample_indices])
                sup_occupancies.extend(boundary_sample_occupancies[subsample_indices])
        else:
            supervision_path = path + '/points.npz'
            sup_points_dict = np.load(supervision_path)
            sup_coords = sup_points_dict['points']
            # Break symmetry if given in float16:
            if sup_coords.dtype == np.float16:
                sup_coords = sup_coords.astype(np.float32)
                sup_coords += 1e-4 * np.random.randn(*sup_coords.shape)
            else:
                sup_coords = sup_coords.astype(np.float32)
            sup_occupancies = sup_points_dict['occupancies']
            sup_occupancies = np.unpackbits(sup_occupancies)[:sup_coords.shape[0]]
            sup_occupancies = sup_occupancies.astype(np.float32)
            sup_subsample_indices = np.random.randint(0, len(sup_coords), self.nsamples_supervision)
            sup_coords = sup_coords[sup_subsample_indices]
            sup_occupancies = sup_occupancies[sup_subsample_indices]

        assert len(sup_coords) == self.nsamples_supervision
        assert len(sup_occupancies) == self.nsamples_supervision

        sup_coords = np.array(sup_coords, dtype=np.float32)
        sup_occupancies = np.array(sup_occupancies, dtype=np.float32)


        # additive gaussian noise
        if self.noise_std is not None:
            obs = obs + float(self.noise_std) * np.random.randn(obs.shape[0], obs.shape[1]).astype(np.float32)

        # for IF-Net the observed input point cloud needs to be discretized to a voxel grid
        if self.voxelize:
            input_occupancies = np.zeros(len(self.grid_points), dtype=np.int8)
            _, idx = self.kdtree.query(obs)
            input_pc = np.asarray(obs, dtype=np.float32)
            input_occupancies[idx] = 1
            input = np.reshape(input_occupancies, (self.res,) * 3)
            obs = np.array(input, dtype=np.float32)
            #sup_coords = np.array(sup_coords_vox, dtype=np.float32)
            return {'occupancies': sup_occupancies.astype(np.float32), 'points': sup_coords.astype(np.float32), # TODO change name of points to be more explicit?
                    'inputs': obs.astype(np.float32), 'input_pc': input_pc, 'path': path}
        return {'occupancies': sup_occupancies.astype(np.float32), 'points': sup_coords.astype(np.float32), # TODO change name of points to be more explicit?
                'inputs': obs.astype(np.float32), 'path': path}

    # TODO: Not sure what I am doing here
    def get_loader(self, shuffle =True):
        if self.mode == 'test':
            random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            np.random.seed(1)
            return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        else:
            return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


def get_shapenet_dataset(mode, dataset_type, CFG):
    pcsamples = CFG['training']['pc_samples']
    npoints_decoder = CFG['training']['npoints_decoder']
    batch_size = CFG['training']['batch_size']
    noisy_obs = CFG['mode']['noisy_obs']
    vox = CFG['encoder']['type'] == 'ifnet'
    if dataset_type == 'onet':
        data_path = 'occupancy_networks/data/ShapeNet',  # TODO
    else:
        data_path = 'shapenet/data'
    return ShapeReconstructionDataset(mode, dataset_type, data_path,
                                      nsamples_input=pcsamples,
                                      nsamples_supervision=npoints_decoder,
                                      batch_size=batch_size, num_workers=10,
                                      noise_std=noisy_obs,
                                      voxelize=vox
                                      )

# Treated as separate class, since this is only used for testing (zero-shot generalization)
class HumanDataset(Dataset):
    """
    Datasets that loads point clouds sampled from the surface of an object.

    Attributes:
        mode str: either 'train', 'val' or 'test', determines from which split to sample data from
        nsamples_input int: number of points in input point cloud
        nsamples_supervision int: number of supervision points i.e. points for which loss is calculated
        single_view bool: when False, points are sampled uniformly on the surface, when True, object is observed from a
            single view point from the upper hemisphere
        test_equiv bool: False when training, when set to True input is randomly rotated in order to investigate
            rotation equivariance of model.
    """

    def __init__(self, nsamples_input = 3000,
                 batch_size = 64, nsamples_supervision = 1024, num_workers = 10, voxelize=False):

        self.path = 'MPI-FAUST/test/scans/' #TODO
        self.data = ['test_scan_{:03d}.ply'.format(i) for i in range(200)]

        self.nsamples_supervision = nsamples_supervision
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nsamples_input = nsamples_input
        self.voxelize = voxelize
        if voxelize:
            self.res = 128
            self.grid_points = create_grid_points_from_bounds(-0.5, 0.5, self.res)
            self.kdtree = KDTree(self.grid_points)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        input_points = trimesh.load(path).vertices.__array__().astype(np.float32)

        vmax = np.max(input_points)
        vmin = np.min(input_points)

        bbox_length = vmax - vmin
        bbox_middle = (vmax + vmin) / 2
        input_points -= bbox_middle
        input_points /= bbox_length

        input_idx = np.random.randint(0, input_points.shape[0], self.nsamples_input)
        obs = input_points[input_idx]

        if self.voxelize:
            input_occupancies = np.zeros(len(self.grid_points), dtype=np.int8)
            _, idx = self.kdtree.query(obs)
            input_pc = np.asarray(obs, dtype=np.float32)
            input_occupancies[idx] = 1
            input = np.reshape(input_occupancies, (self.res,) * 3)
            obs = np.array(input, dtype=np.float32)
            return {'inputs': obs, 'input_pc': input_pc, 'path': path}

        return {'inputs': obs, 'path': path}

    def get_loader(self, shuffle=True):
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


def get_human_dataset(CFG):
    vox = CFG['encoder']['type'] == 'ifnet'
    return HumanDataset(nsamples_input=CFG['training']['pc_samples'],
                        nsamples_supervision=CFG['training']['npoints_decoder'],
                        batch_size=CFG['training']['batch_size'], num_workers=10, voxelize=vox)
