from src.simulator import run_flower_simulation, run_federated_simulation, init_dataset_and_model, set_seed, ResultManager, get_client_datasets
from src.config import Config
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
import os
import logging
from copy import deepcopy
logger = logging.getLogger('SIMULATOR')
import json
from tests.test_utils import compose_config

# def compose_config() -> Config:
#     initialize_config_dir(version_base=None, config_dir="/home/asim.ukaye/fed_learning/feduciary/conf", job_name="test_app")
#     # print(os.getcwd())
#     if not os.path.exists('test_outputs'):
#         os.makedirs('test_outputs', exist_ok=True)
#     os.chdir('test_outputs')
#     cs = ConfigStore.instance()
#     cs.store('base_config', node=Config)
#     cfg = compose(config_name='config')
#     return OmegaConf.to_object(cfg) #type: ignore

def set_test_params(cfg: Config):
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = False
    cfg.simulator.num_rounds = 2

    cfg.client.cfg.epochs = 1
    cfg.client.cfg.metric_cfg.log_to_file = True

    cfg.simulator.num_clients = 2
    cfg.dataset.split_conf.num_splits = 2

    cfg.dataset.subsample = True
    cfg.dataset.subsample_fraction = 0.05
    cfg.dataset.split_conf.split_type = 'iid'

    return cfg


def test_flower_native_match():
    cfg = compose_config()
    cfg = set_test_params(cfg)
    assert isinstance(cfg, Config)

    test, train, model = init_dataset_and_model(cfg)
    set_seed(cfg.simulator.seed)

    flower_cfg = deepcopy(cfg)
    fed_cfg = deepcopy(cfg)

    flower_cfg.simulator.out_prefix = 'flower_'
    fed_cfg.simulator.out_prefix = 'fed_'

    flower_cfg.client.cfg.metric_cfg.file_prefix = 'flower_'
    fed_cfg.client.cfg.metric_cfg.file_prefix = 'fed_'

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


    # print(result_feduciary)
    # print(result_flower)
    assert result_flower.keys() == result_feduciary.keys()

    for value1, value2 in zip(result_flower.values(), result_feduciary.values()):
        if isinstance(value1, dict) and isinstance(value2, dict):
            assert value1.keys() ==value2.keys()
            

        assert value1 ==value2

    assert result_flower == result_feduciary