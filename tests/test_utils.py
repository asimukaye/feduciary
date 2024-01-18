from feduciary.config import Config, register_configs
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
import os


def compose_config() -> Config:
    if not os.path.exists('test_outputs'):
        os.makedirs('test_outputs', exist_ok=True)
    os.chdir('test_outputs')
    # gh =GlobalHydra.instance()
    # gh.clear()

    with initialize(version_base=None, config_path="../conf", job_name="test_app"):
    # print(os.getcwd())
        # cs = ConfigStore.instance()
        # cs.store('base_config', node=Config)
        register_configs()
        cfg = compose(config_name='config')
    return OmegaConf.to_object(cfg) #type: ignore