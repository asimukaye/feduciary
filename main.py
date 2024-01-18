import hydra
from omegaconf import OmegaConf
from feduciary.simulator import Simulator
from feduciary.config import Config, register_configs
import logging
import torch.multiprocessing as torch_mp
import cProfile, pstats

# Icecream debugger
from icecream import install, ic
install()
ic.configureOutput(includeContext=True)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
# @hydra.main(version_base=None, config_path="conf", config_name="flowerconfig")
def run_feduciary(cfg: Config):

    cfg_obj: Config = OmegaConf.to_object(cfg)
    logger.debug((OmegaConf.to_yaml(cfg)))
    # logger.debug(cfg_obj.dataset.split_conf.__dict__)
    # logger.debug(cfg_obj.client.cfg.__dict__)


    sim = Simulator(cfg_obj)
    sim.run_simulation()
    sim.finalize()

if __name__== "__main__":
    torch_mp.set_start_method('spawn')
    register_configs()
    # cProfile.run('run_feduciary()', 'feduciary_stats', pstats.SortKey.CUMULATIVE)
    run_feduciary()
