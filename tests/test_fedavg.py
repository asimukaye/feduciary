import os
from copy import deepcopy

from feduciary.config import Config, register_configs
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from test_utils import compose_config
from feduciary.simulator import set_seed
from flwr.server.strategy import FedAvg
from feduciary.data import load_federated_dataset
from feduciary.strategy import FedAvgStrategy
from feduciary.simulator import run_flower_simulation, run_federated_simulation, init_dataset_and_model, set_seed, ResultManager, get_client_datasets



def set_test_params(cfg: Config):
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = False
    cfg.simulator.num_rounds = 10

    cfg.client.train_cfg.epochs = 1
    cfg.client.train_cfg.metric_cfg.log_to_file = True

    cfg.simulator.num_clients = 6
    cfg.dataset.split_conf.num_splits = 6

    cfg.dataset.subsample = True
    cfg.dataset.subsample_fraction = 0.05
    cfg.dataset.split_conf.split_type = 'iid'

    return cfg

def test_fedavg():
    cfg = compose_config()
    cfg = set_test_params(cfg)
    
    set_seed(cfg.simulator.seed)


    flower_cfg = deepcopy(cfg)
    fed_cfg = deepcopy(cfg)

    flower_cfg.simulator.out_prefix = 'flower_'
    fed_cfg.simulator.out_prefix = 'fed_'

    flower_cfg.client.train_cfg.metric_cfg.file_prefix = 'flower_'
    fed_cfg.client.train_cfg.metric_cfg.file_prefix = 'fed_'

    flower_cfg.server.train_cfg.metric_cfg.file_prefix = 'flower_'
    fed_cfg.server.train_cfg.metric_cfg.file_prefix = 'fed_'

    flower_model = deepcopy(model)
    fed_model = deepcopy(model)
    run_flower_simulation(cfg=flower_cfg, train_set=train, test_set=test,model=flower_model)
    set_seed(cfg.simulator.seed)

    run_federated_simulation(cfg=fed_cfg, train_set=train, test_set=test, model_instance=fed_model)

    with open('flower_int_result.json', 'r') as flower_json:
        result_flower:dict = json.load(flower_json)
    with open('fed_int_result.json', 'r') as fed_json:
        result_feduciary: dict = json.load(fed_json)


    assert result_flower.keys() == result_feduciary.keys()

    for value1, value2 in zip(result_flower.values(), result_feduciary.values()):
        if isinstance(value1, dict) and isinstance(value2, dict):
            assert value1.keys() ==value2.keys()
            

        assert value1 ==value2

    assert result_flower == result_feduciary
