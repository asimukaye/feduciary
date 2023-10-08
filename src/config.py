# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, MISSING
from typing import Optional, Any
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from torch import cuda
from hydra.utils import to_absolute_path
from src.utils import Range
from torch import cuda
from torch.backends import mps

import logging

logger = logging.getLogger(__name__)

########## Simulator Configurations ##########
@dataclass
class SimConfig:
    seed: int
    num_clients: int
    use_tensorboard: bool
    num_rounds: int
    # def __post_init__(self):
    #     pass

########## Client Configurations ##########

@dataclass
class ClientConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    optimizer: str = field()
    criterion: str = field()
    lr: float = field()
    lr_decay: Optional[float] = field()
    lr_scheduler: Optional[dict] = field()
    lr_decay_step: Optional[int] = field()
    beta: Optional[float] = field()
    shuffle: bool = field()
    eval_metrics: list
    
    def __post_init__(self):
        # if self.device =='cuda':
        #     assert cuda.is_available(), 'Please check if your GPU is available !' 
        assert self.batch_size >= 1

        if cuda.is_available():
            self.device = 'cuda'
        elif mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        logger.info(f'Auto Configured device to: {self.device}')

@dataclass
class ClientSchema:
    _target_: str 
    _partial_: bool
    cfg: ClientConfig 

########## Server Configurations ##########

@dataclass
class ServerConfig:
    eval_type: str  = field(default='both')
    eval_fraction: float  = field(default=1.0)
    eval_every: int  = field(default=1)
    sampling_fraction: float = field(default=1.0)
    rounds: int = 1

    def __post_init__(self):
        assert self.sampling_fraction == Range(0.0, 1.0), f'Invalid value {self.sampling_fraction} for sampling fraction'
        assert self.eval_fraction == Range(0., 1.0)
        assert self.eval_type == 'both' # Remove after 


@dataclass
class CGSVConfig(ServerConfig):
    beta: float = 1.5
    alpha: float = 0.95
    gamma: float = 0.5
    
    def __post_init__(self):
        super().__post_init__()
        

@dataclass
class FedavgConfig(ServerConfig):
    momentum: float = float('nan')

    def __post_init__(self):
        super().__post_init__()
        assert self.momentum >= 0.0

@dataclass
class VarAggConfig(ServerConfig):
    beta: float = 1.5
    alpha: float = 0.95
    gamma: float = 0.5
    
    def __post_init__(self):
        super().__post_init__()
        

@dataclass
class ServerSchema:
    _target_: str
    _partial_: bool 
    cfg: ServerConfig
    client_cfg: ClientConfig


########## Other configurataions ##########

@dataclass
class TransformsConfig:
    resize: Optional[dict] = field(default_factory=dict)
    normalize:Optional[dict] = field(default_factory=dict)
    train_cfg:Optional[list] = field(default_factory=list)

    def __post_init__(self):
        train_tf = []
        for key, val in self.__dict__.items():
            if key == 'normalize':
                continue
            else:
                if val:
                    train_tf.append(val)
        self.train_cfg = train_tf  
    # construct

@dataclass
class DatasetConfig:
    name: str
    data_path: str
    split_type: str
    test_fraction: float
    K: int  # num_clients
    transforms: Optional[TransformsConfig]
    subsample: float = 0.0  # subsample the dataset with the given fraction
    # rawsmpl: Optional[int]
    # use_model_tokenizer: Optional[bool]
    # use_pt_model: Optional[bool]


    def __post_init__(self):
        assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        self.data_path = to_absolute_path(self.data_path)

@dataclass
class ModelSpecConfig:
    _target_: str 
    _partial_: bool 
    hidden_size: Optional[int]  
    num_classes: Optional[int] 
    in_channels: Optional[int] 

@dataclass
class ModelConfig:
    name: str
    init_type: str
    init_gain: float
    dropout: float
    model_spec: ModelSpecConfig


# TODO: Build on this later and develop into a unified evaluation module
# @dataclass
# class MetricConfig:
#     eval_metrics: list
#     fairness_metrics: list

# modes 
#  possible enum debug
########## Master Configurations ##########
@dataclass
class Config():
    mode: str = field()
    simulator: SimConfig = field(default=SimConfig)
    server: ServerSchema = field(default=ServerSchema)
    client: ClientSchema = field(default=ClientSchema)
    dataset: DatasetConfig = field(default=DatasetConfig)
    model: ModelConfig = field(default=ModelConfig)
    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):
        # if self.dataset.use_model_tokenizer or self.dataset.use_pt_model:
        #     assert self.model.name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'
        if self.mode == 'debug':
            set_debug_mode(self)



def set_debug_mode(cfg: Config):
    # if cuda.is_available():
    #     cfg.client.cfg.device = 'cuda'
    # elif mps.is_available():
    #     cfg.client.cfg.device = 'mps'
    # else:
    #     cfg.client.cfg.device = 'cpu'
    
    logger.setLevel(logging.DEBUG)
    logger.debug(f'Setting device to: {cfg.client.cfg.device}')

    cfg.simulator.num_rounds = 2
    logger.debug(f'Setting rounds to: {cfg.simulator.num_rounds}')

    cfg.dataset.subsample = 0.2



def register_configs():
    cs = ConfigStore.instance()
    cs.store(name='base_config', node=Config)
    cs.store(group='server', name='base_server', node=ServerSchema)
    cs.store(group='server/cfg', name='base_cgsv', node=CGSVConfig)
    cs.store(group='server/cfg', name='base_fedavg', node=FedavgConfig)

