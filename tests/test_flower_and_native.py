from src.simulator import run_flower_simulation, run_federated_simulation, init_dataset_and_model, set_seed, ResultManager
from src.config import Config
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
import os
import logging
from copy import deepcopy
logger = logging.getLogger('SIMULATOR')

def compose_config() -> Config:
    initialize_config_dir(version_base=None, config_dir="/home/asim.ukaye/fed_learning/feduciary/conf", job_name="test_app")
    print(os.getcwd())
    cs = ConfigStore.instance()
    cs.store('base_config', node=Config)
    cfg = compose(config_name='config')
    return OmegaConf.to_object(cfg) #type: ignore

def set_test_params(cfg: Config):
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = False
    cfg.simulator.num_rounds = 2

    cfg.client.cfg.epochs = 1

    cfg.simulator.num_clients = 2
    cfg.dataset.split_conf.num_splits = 2

    cfg.dataset.subsample = True
    cfg.dataset.subsample_fraction = 0.05

    return cfg


def test_flower_native_match():
    cfg = compose_config()
    cfg = set_test_params(cfg)
    assert isinstance(cfg, Config)

    test, train, model = init_dataset_and_model(cfg)

    res_flower = ResultManager(cfg.simulator, logger)

    res_feduciary = ResultManager(cfg.simulator, logger)

    run_flower_simulation(cfg=deepcopy(cfg), train_set=train, test_set=test,model=model, result_manager=res_flower)

    run_federated_simulation(cfg=deepcopy(cfg), train_set=train, test_set=test, model_instance=model, result_manager=res_feduciary)

    result_flower = res_flower.last_result

    result_feduciary = res_feduciary.last_result

    print(result_feduciary)
    print(result_flower)

    assert result_flower == result_feduciary