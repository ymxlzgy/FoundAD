# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import logging
import multiprocessing as mp

import pprint
import yaml
import torch

from src.train import main as app_main_mvtec
# from src.evaluation import main as app_eval
from src.AD import main as AD

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--mode', type=str,
    help='train or AD',
    default='train')

def process_main(rank, fname, world_size, devices, mode):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(40112)
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank)

    if mode == 'train':
        app_main_mvtec(args=params)
    elif mode =='AD':
        AD(args=params)

if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.mode)
        ).start()
