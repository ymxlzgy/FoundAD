# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import logging
import multiprocessing as mp

import pprint
import yaml
import torch

from src.train import main as app_main_mvtec
# from src.evaluation import main as app_eval
from src.AD import main as AD

import hydra
from omegaconf import DictConfig, OmegaConf

# def process_main(rank, fname, world_size, devices, mode):
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
#     logging.basicConfig()
#     logger = logging.getLogger()
#     if rank == 0:
#         logger.setLevel(logging.INFO)
#     else:
#         logger.setLevel(logging.ERROR)

#     logger.info(f'called-params {fname}')

#     # -- load script params
#     params = None
#     with open(fname, 'r') as y_file:
#         params = yaml.load(y_file, Loader=yaml.FullLoader)
#         logger.info('loaded params...')
#         pp = pprint.PrettyPrinter(indent=4)
#         pp.pprint(params)

#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = str(40112)
#     torch.distributed.init_process_group(
#         backend='nccl',
#         world_size=world_size,
#         rank=rank)

#     if mode == 'train':
#         app_main_mvtec(args=params)
#     elif mode =='AD':
#         AD(args=params)

def process_main(rank: int, cfg_dict: dict, world_size: int):
    """
    rank: local rank
    cfg_dict: dict config
    world_size: len(devices)
    """
    # ----
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    devices = cfg_dict.get("devices", ["cuda:0"])
    mode = cfg_dict.get("mode", "train")
    dist = cfg_dict.get("dist", {})
    master_addr = dist.get("master_addr", "localhost")
    master_port = str(dist.get("master_port", 40112))
    backend = dist.get("backend", "nccl")

    dev_str = devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_str.split(":")[-1])

    model_params = cfg_dict.get("app", {}) # model config

    model_params.update(cfg_dict)

    logger.info("model params:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(model_params)

    # ---- DDP    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank
    )

    if rank==0:
        log_dir = model_params["logging"]["folder"]
        os.makedirs(log_dir, exist_ok=True)
        params_save_path = os.path.join(log_dir, "params.yaml")
        with open(params_save_path, "w") as f:
            yaml.safe_dump(model_params, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {params_save_path}")

    if mode == "train":
        app_main_mvtec(args=model_params)
    elif mode == "AD":
        AD(args=model_params)
    else:
        if rank == 0:
            logger.error(f"Unknown mode: {mode}")

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    devices = cfg_dict.get("devices", ["cuda:0"])
    world_size = len(devices)

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=process_main, args=(rank, cfg_dict, world_size))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == '__main__':
    main()