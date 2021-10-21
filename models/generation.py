import mcubes
import trimesh
import torch
import os
from glob import glob
import numpy as np
from models.utils import libmise, libmcubes, libsimplify


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


class Generator(object):
    def __init__(self, encoder, decoder, threshold, exp_name, checkpoint=None, path=None, device=torch.device("cuda"),
                 resolution=128, batch_points=1000000, is_IF=False, method='mise'):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.decoder = decoder.to(device)
        self.decoder.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.path = path
        self.checkpoint_dir = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(exp_name)
        self.is_IF = is_IF
        self.load_checkpoint(checkpoint)
        self.batch_points = batch_points

        self.min = -0.501
        self.max = 0.501

        assert method in ['mise', 'mcubes']
        self.method = method

        if method == 'mcubes':
            self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.resolution)
            self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
            self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)
                #self.grid_points_split = torch.split(grid_points, self.batch_points, dim=1)


    def get_logits_MISE(self, encoding):
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        mesh_extractor = libmise.MISE(self.resolution[0], self.resolution[1], threshold)
        box_size = 1 + 0.02  # 1 + self.padding
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = points / mesh_extractor.resolution
            # Normalize to bounding box
            pointsf = box_size * (pointsf - 0.5)
            #if self.is_IF:
            #    p_copy = pointsf.copy()
            #    pointsf[:, 0], pointsf[:, 2] = p_copy[:, 2], p_copy[:, 0]
            #    pointsf = 2 * pointsf
            pointsf = torch.FloatTensor(pointsf).to(self.device).unsqueeze(0)

            points_split = torch.split(pointsf, self.batch_points, dim=1)
            # Evaluate model and update
            occ_hat = []
            with torch.no_grad():
                for p in points_split:
                    occ_hat.append(self.decoder(p, encoding).squeeze()) # TODO change to encoder for IFNET!
            values = torch.cat(occ_hat, dim=0).cpu().numpy()
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        value_grid = mesh_extractor.to_dense()
        return value_grid

    def get_logits_mcubes(self, encoding):
        sample_points = self.grid_points.clone()
        if self.is_IF:
            boundary_sample_coords = sample_points.clone()
            sample_points[:, :, 0], sample_points[:, :, 2] = boundary_sample_coords[:, :, 2], boundary_sample_coords[:, :, 0]

        grid_points_split = torch.split(sample_points, self.batch_points, dim=1)
        logits_list = []
        for points in grid_points_split:
            with torch.no_grad():
                    logits = self.decoder(points, encoding).squeeze()
                    logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()
        return logits

    def generate_mesh(self, data):
        inputs = data['inputs'].to(self.device)

        with torch.no_grad():
            encoding = self.encoder(inputs)

        if self.method == 'mcubes':
            logits = self.get_logits_mcubes(encoding)
        else:
            logits = self.get_logits_MISE(encoding)
        return logits

    def mesh_from_logits_mcubes(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=-1000)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def mesh_from_logits_MISE(self, occ_hat):
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + 0.02
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh = libsimplify.simplify_mesh(mesh, 100000, 5.)
        return mesh

    def load_checkpoint(self, checkpoint=None, path=None):
        if checkpoint is None and self.path is None:
            checkpoints = glob(self.checkpoint_dir + '/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_dir))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_dir + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        elif checkpoint is None:
            path = self.path
        else:
            path = self.checkpoint_dir + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path, map_location=self.device)
        if self.is_IF:
            # bc. the model was trained as a whole all parameters are present in the encoder state dict (the only state dict) #todo
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])