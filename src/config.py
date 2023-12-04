# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, asdict
from typing import Optional
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from torch import cuda
from hydra.utils import to_absolute_path, get_object
from src.utils import Range
from torch import cuda
from torch.backends import mps
from pandas import json_normalize
import logging
import inspect
logger = logging.getLogger(__name__)

def arg_check(args:dict, fn:str =None):
    # Figure out usage with string functions
    # Check if the argument spec is compatible with
    if fn is None:
        fn = args['_target_']

    all_args = inspect.signature(get_object(fn)).parameters.values()
    required_args = [arg.name for arg in all_args if arg.default==inspect.Parameter.empty]
    # collect eneterd arguments
    for argument in required_args:
        if argument in args:
            logger.debug(f'Found required argument: {argument}')
        else:
            if args.get('_partial_', False):
                logger.debug(f'Missing required argument: {argument} for {fn}')
            else:
                logger.error(f'Missing required argument: {argument}')
                raise ValueError(f'Missing required argument: {argument}')

########## Simulator Configurations ##########
@dataclass
class SimConfig:
    seed: int
    num_clients: int
    use_tensorboard: bool
    num_rounds: int
    use_wandb: bool
    checkpoint_every: int = field(default=10)
    mode: str = field(default='federated')

    def __post_init__(self):
        assert self.mode in ['federated', 'standalone', 'centralized'], f'Unknown simulator mode: {self.mode}'


########## Client Configurations ##########

@dataclass
class ClientConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    optimizer: dict = field()
    criterion: dict = field()
    lr: float = field()
    lr_decay: Optional[float] = field()
    lr_scheduler: Optional[dict] = field()
    shuffle: bool = field(default=False)
    eval_metrics: list = field(default_factory=list)
    
    def __post_init__(self):
        # if self.device =='cuda':
        #     assert cuda.is_available(), 'Please check if your GPU is available !' 
        assert self.batch_size >= 1
        if self.device == 'auto':
            if cuda.is_available():
                self.device = 'cuda'
            elif mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            logger.info(f'Auto Configured device to: {self.device}')
        arg_check(self.lr_scheduler)
        arg_check(self.optimizer)


def default_seed():
    return [1,2,3]
@dataclass
class VaraggClientConfig(ClientConfig):
    seeds: list[int] = field(default_factory=default_seed)
    
    def __post_init__(self):
        super().__post_init__()

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
    multiprocessing:bool = False

    def __post_init__(self):
        assert self.sampling_fraction == Range(0.0, 1.0), f'Invalid value {self.sampling_fraction} for sampling fraction'
        assert self.eval_fraction == Range(0., 1.0)
        assert self.eval_type == 'both' # Remove later


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
class VaraggServerConfig(ServerConfig):
    alpha: float = 0.95
    gamma: float = 0.5
    betas: list = field(default_factory=list)
    weight_scaling: str = field(default='tanh')
    
    def __post_init__(self):
        super().__post_init__()
        assert self.weight_scaling in ['tanh', 'min_max'], 'Incorrect weight scaling type'
        

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
class NoiseConfig:
    mu: float = 0.0
    sigma: float = 1.0
    flip_percent: float = 0.5
    
@dataclass
class DatasetConfig:
    name: str
    data_path: str
    split_type: str
    test_fraction: float
    num_clients: int  # num_clients
    noise: Optional[NoiseConfig]
    transforms: Optional[TransformsConfig]
    subsample: float = 0.0  # subsample the dataset with the given fraction


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
    desc: str = field()
    log_conf: list = field(default_factory=list)
    simulator: SimConfig = field(default=SimConfig)
    server: ServerSchema = field(default=ServerSchema)
    client: ClientSchema = field(default=ClientSchema)
    dataset: DatasetConfig = field(default=DatasetConfig)
    model: ModelConfig = field(default=ModelConfig)
    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):
        # if self.dataset.use_model_tokenizer or self.dataset.use_pt_model:
        #     assert self.model.name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'
        flat_cfg = json_normalize(asdict(self))
        if not all(arg in flat_cfg for arg in self.log_conf):
            raise(KeyError(f'Recheck the keys set in log_conf: {self.log_conf}'))
        if self.mode == 'debug':
            set_debug_mode(self)



def set_debug_mode(cfg: Config):

    logger.root.setLevel(logging.DEBUG)
    cfg.simulator.use_wandb = False
    logger.debug(f'[Debug Override] Setting use_wandb to: {cfg.simulator.use_wandb}')
    cfg.simulator.num_rounds = 2
    logger.debug(f'[Debug Override] Setting rounds to: {cfg.simulator.num_rounds}')
    cfg.client.cfg.epochs = 1
    logger.debug(f'[Debug Override] Setting epochs to: {cfg.client.cfg.epochs}')

    cfg.simulator.num_clients = 3
    cfg.dataset.K = 3
    logger.debug(f'[Debug Override] Setting num clients to: {cfg.simulator.num_clients}')


    cfg.dataset.subsample = 0.01


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name='base_config', node=Config)
    cs.store(group='client', name='client_schema', node=ClientSchema)
    cs.store(group='client/cfg', name='base_client', node=ClientConfig)
    cs.store(group='server', name='base_server', node=ServerSchema)
    cs.store(group='server/cfg', name='base_cgsv', node=CGSVConfig)
    cs.store(group='server/cfg', name='base_fedavg', node=FedavgConfig)
    cs.store(group='server/cfg', name='varagg_server', node=VaraggServerConfig)
    cs.store(group='client/cfg', name='varagg_client', node=VaraggClientConfig)

