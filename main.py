import os 
import yaml
from typing import Any, Optional
from torch.utils.tensorboard import SummaryWriter

# from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model 

import torch
from dataclasses import dataclass, field
# from dataclasses_wiz
# from attrs import define, field
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from hydra.utils import instantiate
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimConfig:
    seed: int
    exp_prefix: str
    data_path: str
    result_path: str
    num_clients: int
    eval_type: str
    eval_fraction: float
    eval_every: int
    use_tensorboard: bool
    device: str

@dataclass
class CGSVConfig:
    name: str
    beta: float
    alpha: float
    gamma: float = field()

@dataclass
class CGSVConfig:
    name: str
    beta: float
    alpha: float
    gamma: float = field()

@dataclass
class ClientConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    optimizer: str = field()
    criterion: str = field()
    lr: float = field()
    lr_decay: Optional[float] = field()
    no_shuffle: bool = field()
    
    @optimizer.validator
    def _check_optimizer(self, attribute, value):
        assert value in torch.optim.__dict__.keys(), f"Invalid {attribute.name} config: {value}"
    @criterion.validator
    def _check_criterion(self, attribute, value):
        assert value in torch.nn.__dict__.keys(), f"Invalid {attribute.name} config: {value}"


@dataclass
class ServerConfig:
    algorithm: CGSVConfig = field()
    sampling_fraction: float = field(default=1.0)
    rounds: int = 1

    @sampling_fraction.validator
    def _range_check(self, attribute, value):
        if value > 1.0 or value<0.0:
            raise ValueError(f'Invalid value: {value} for sampling fraction. It should be in range [0,1]')

@dataclass
class DatasetConfig:
    name: str
    resize: int
    split_type: str

@dataclass
class MetricConfig:
    eval_metrics: list
    fairness_metrics: list

@dataclass
class ModelConfig:
    name: str
    init_type: str
    init_gain: float
    dropout: float

@dataclass
class Config():
    simulator: SimConfig = field(default_factory=SimConfig)
    server: ServerConfig = field(default=ServerConfig)
    client: ClientConfig = field(default=ClientConfig)
    dataset: DatasetConfig = field(default=DatasetConfig)
    metrics: MetricConfig = field(default=MetricConfig)
    model: ModelConfig = field(default=ModelConfig)


cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
# cs.store(group='algorithm', name='base_cgsv', node=CGSVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_loop(cfg: Config):

    print(type(cfg.server.algorithm))
    print(cfg.keys())
    # cfg_obj = instantiate(cfg, object=Config)
    cfg_obj: Config = OmegaConf.to_object(cfg)
    # print(cfg_obj)
    print(type(cfg_obj))

    # logger.info(f'Done configureing {cfg_obj}')
    # print(type(cfg_obj))
    # schema = OmegaConf.structured(Config)
    
    # print(type(schema))
    # print(OmegaConf.merge(schema, cfg))

    
if __name__== "__main__":

    main_loop()

    
