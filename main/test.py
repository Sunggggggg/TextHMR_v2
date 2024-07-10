import argparse
import __init_path
from tqdm import tqdm

from core.base import get_dataloader
from core.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)

dataset_names = ['Human36M', 'MPII3D', 'PW3D']
is_train = True
dataset_list, dataloader_list = get_dataloader(args, dataset_names, is_train)

batch_generator = tqdm(dataloader_list)
for i, (inputs, targets, meta) in enumerate(batch_generator):
    print()

    break 