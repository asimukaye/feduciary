from src.config import Config
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir, _internal
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
import os


def compose_config() -> Config:
    gh =GlobalHydra.instance()
    gh.clear()
    initialize_config_dir(version_base=None, config_dir="/home/asim.ukaye/fed_learning/feduciary/conf", job_name="test_app")
    # print(os.getcwd())
    if not os.path.exists('test_outputs'):
        os.makedirs('test_outputs', exist_ok=True)
    os.chdir('test_outputs')
    cs = ConfigStore.instance()
    cs.store('base_config', node=Config)
    cfg = compose(config_name='config')
    return OmegaConf.to_object(cfg) #type: ignore