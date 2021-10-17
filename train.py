import data.ShapeNetDatasets as shapenet_data
from models import training
import argparse
import os, yaml
import torch

from models.build_model import build_model

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
#TODO incorporate in yaml
parser.add_argument('-interp_dec', action='store_true')
parser.set_defaults(interp_dec=False)
parser.add_argument('-pointnet_enc', action='store_true')
parser.set_defaults(pointnet_enc=False)

parser.add_argument('-pc_samples' , default=300, type=int)
parser.add_argument('-batch_size' , default=64, type=int)
parser.add_argument('-cuda_device', default=0, type=int)
parser.add_argument('-data_type', required=True, type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]


exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    assert args.cfg_file is not None
    CFG = yaml.safe_load(open(args.cfg_file, 'r'))
    print('Creating checkpoint dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)


print(CFG)

torch.cuda.set_device(args.cuda_device)

encoder, decoder = build_model(CFG)
encoder.float()
decoder.float()

train_dataset = shapenet_data.get_shapenet_dataset('train', args.data_type, CFG)
val_dataset = shapenet_data.get_shapenet_dataset('val', args.data_type, CFG)

device = torch.device("cuda")


train_cfg = CFG['training']
trainer = training.Trainer(encoder, decoder, train_cfg, device, train_dataset, val_dataset, args.exp_name,
                           optimizer='Adam')
trainer.train_model(1500)
