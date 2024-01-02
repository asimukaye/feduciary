import hydra
from omegaconf import OmegaConf
from src.simulator import Simulator
from src.config import Config, register_configs
import logging
import torch.multiprocessing as torch_mp

# Icecream debugger
from icecream import install, ic
install()
ic.configureOutput(includeContext=True)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
# @hydra.main(version_base=None, config_path="conf", config_name="flowerconfig")
def run_feduciary(cfg: Config):

    cfg_obj: Config = OmegaConf.to_object(cfg)
    logger.debug((OmegaConf.to_yaml(cfg_obj)))

    sim = Simulator(cfg_obj)
    sim.run_simulation()
    sim.finalize()

if __name__== "__main__":
    torch_mp.set_start_method('spawn')
    register_configs()
    run_feduciary()
