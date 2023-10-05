
from typing import Any, Optional
# from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import OmegaConf
from src.simulator import Simulator
from src.config import Config, register_configs
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_feduciary(cfg):
    cfg_obj: Config = OmegaConf.to_object(cfg)
    # print(type(cfg_obj))
    # print(cfg_obj)

    logger.info((OmegaConf.to_yaml(cfg_obj)))

    # exit(0)
    sim = Simulator(cfg_obj)
    sim.run_simulation()
    sim.finalize()

if __name__== "__main__":
    register_configs()
    run_feduciary()

    
