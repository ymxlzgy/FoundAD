import os
import logging
import multiprocessing as mp

import pprint
import yaml
import torch

from src.train import main as app_main_mvtec
from src.AD import main as AD, _demo

import hydra
from omegaconf import DictConfig, OmegaConf


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

    params = cfg_dict.get("app", {}) # model config

    params.update(cfg_dict)

    logger.info("Params:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(params)

    # ---- DDP    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank
    )

    if mode == "train":
        if rank==0:
            log_dir = params["logging"]["folder"]
            os.makedirs(log_dir, exist_ok=True)
            params_save_path = os.path.join(log_dir, "params.yaml")
            with open(params_save_path, "w") as f:
                yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)
            print(f"Config saved to {params_save_path}")

        app_main_mvtec(args=params)
    elif mode == "AD":
        load_path = os.path.join('logs', cfg_dict['data']['data_name'], params.get('model_name','')+cfg_dict['diy_name'])
        saved_path = os.path.join(load_path,"params.yaml")
        if os.path.exists(saved_path):
            with open(saved_path, "r") as f:
                saved_params = yaml.safe_load(f)
                assert cfg_dict['diy_name']==saved_params['diy_name']
                params['meta'] = saved_params['meta']
                params["ckpt_path"] = os.path.join(saved_params["logging"]["folder"],f"train-step{params['ckpt_step']}.pth.tar")
                params["logging"]["folder"] = os.path.join(saved_params["logging"]["folder"],f"eval/{str(params['ckpt_step'])}")
            AD(args=params)
        else:
            print("No ckpt path is found.")
    elif mode == "demo":
        load_path = os.path.join('logs', cfg_dict['data']['data_name'], params.get('model_name','')+cfg_dict['diy_name'])
        saved_path = os.path.join(load_path,"params.yaml")
        if os.path.exists(saved_path):
            with open(saved_path, "r") as f:
                saved_params = yaml.safe_load(f)
                assert cfg_dict['diy_name']==saved_params['diy_name']
                params['meta'] = saved_params['meta']
                params["ckpt_path"] = os.path.join(saved_params["logging"]["folder"],f"pretrained.pth.tar")
                params["logging"]["folder"] = os.path.join(saved_params["logging"]["folder"],"demo")
            print(f"loading {params['ckpt_path']}...")
            _demo(params["ckpt_path"], params)
        else:
            print("No ckpt is found.")
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