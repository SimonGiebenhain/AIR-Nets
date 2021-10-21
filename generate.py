import data.ShapeNetDatasets as shapenet_data
import argparse
from models.generation import Generator
from generation_iterator import gen_iterator
import json, yaml
import torch

from models.build_model import build_model

# python generate.py -exp_name pc3000 -retrieval_res 200 -pc_samples 3000 -checkpoint 228
parser = argparse.ArgumentParser(
    description='Run generation'
)

parser.add_argument('-mode', default='test', type=str)
parser.add_argument('-method', default='mise', type=str)
parser.add_argument('-mcubes_res', default=256, type=int)
parser.add_argument('-mise_res', default=64, type=int)
parser.add_argument('-mise_steps', default=2, type=int)
parser.add_argument('-checkpoint', type=int)
parser.add_argument('-ckpt_name', type=str)
parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cuda_device', default=0, type=int)
parser.add_argument('-data_type', required=True, type=str)

parser.add_argument('-batch_points', default=400000, type=int)

args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.checkpoint is not None or args.ckpt_name is not None

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
with open(fname, 'r') as f:
    print('Loading config file from: ' + fname)
    CFG = yaml.safe_load(f)


print(CFG)
torch.cuda.set_device(args.cuda_device)

encoder, decoder = build_model(CFG)
encoder.float()
decoder.float()

CFG['training']['batch_size'] = 1
if args.data_type in ['ifnet', 'onet']:
    test_dataset = shapenet_data.get_shapenet_dataset('test', args.data_type, CFG)
elif args.data_type == 'human':
    test_dataset = shapenet_data.get_human_dataset(CFG)
else:
    raise ValueError('Unknown data type {}'.format(args.data_type))

is_IF = CFG['decoder']['type'] == 'ifnet'
is_ConvO = CFG['decoder']['type'] == 'convonet'

if args.method not in ['mcubes', 'mise']:
    raise ValueError('Isosurfaces extraction method ' + args.method + ' not known. Has to be one of: "mise", "mcubes"!')
if args.method == 'mcubes':
    retrieval_specs = args.mcubes_res
    retrieval_sepcs_str = str(retrieval_specs)
else:
    retrieval_specs = (args.mise_res, args.mise_steps)
    retrieval_specs_str = str(retrieval_specs[0]) + 'x' + str(retrieval_specs[1])
gen = Generator(encoder, decoder, 0.5, args.exp_name, checkpoint=args.checkpoint, ckpt_name=args.ckpt_name, resolution=retrieval_specs,
                    batch_points=args.batch_points, is_IF=is_IF, method=args.method)

out_path = 'experiments/{}/evaluation_{}_@{}'.format(args.exp_name, args.checkpoint, retrieval_specs_str)

gen_iterator(out_path, test_dataset, gen)