from feduciary.simulator import init_dataset_and_model, set_seed, ResultManager, get_client_datasets
from feduciary.config import Config
import logging
logger = logging.getLogger('TEST_DATA')

from tests.test_utils import compose_config
import numpy as np


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

def test_repeated_data_splits_match():
    cfg = compose_config()
    set_seed(cfg.simulator.seed)
    cfg.dataset.split_conf.num_splits = 3
    print(np.random.get_state())
    test, train, model = init_dataset_and_model(cfg)
    print(np.random.get_state())
    set_seed(cfg.simulator.seed)
    client_datasets_1 = get_client_datasets(cfg.dataset.split_conf, train)
    set_seed(cfg.simulator.seed)
    client_datasets_2 = get_client_datasets(cfg.dataset.split_conf, train)

    assert len(client_datasets_1) == 3
    assert len(client_datasets_2) == 3

    assert len(client_datasets_1[0][0]) == len(client_datasets_1[0][0])

    # print()

    assert client_datasets_1[0][1].indices == client_datasets_2[0][1].indices
    assert client_datasets_1[0][0].indices == client_datasets_2[0][0].indices

    assert client_datasets_1[1][0].indices == client_datasets_2[1][0].indices

    assert client_datasets_1[1][0].indices == client_datasets_2[1][0].indices

    assert client_datasets_1[2][0].indices == client_datasets_2[2][0].indices

