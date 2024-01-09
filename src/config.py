# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import subprocess
from io import StringIO
from abc import ABC
import pandas as pd
from hydra.core.config_store import ConfigStore
from torch import cuda
from hydra.utils import to_absolute_path, get_object
from src.common.utils import Range
from torch import cuda
from torch.backends import mps
from pandas import json_normalize
import logging
import inspect
logger = logging.getLogger(__name__)

def arg_check(args: dict, fn:str|None = None):
    # Figure out usage with string functions
    # Check if the argument spec is compatible with
    if fn is None:
        fn: str = args['_target_']

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

def get_free_gpus(min_memory_reqd= 4096):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    # min_memory_reqd = 10000
    ids = gpu_df.index[gpu_df['memory.free']>min_memory_reqd]
    for id in ids:
        logger.debug('Returning GPU:{} with {} free MiB'.format(id, gpu_df.iloc[id]['memory.free']))
    return ids.to_list()


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    logger.debug('Returning GPU:{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx

########## Simulator Configurations ##########

# @dataclass
# class FlowerConfig:
#     client_resources: dict
def default_resources():
    return {"num_cpus":1, "num_gpus":0.0}


# TODO: Isolate result manager configs from this
@dataclass
class SimConfig:
    seed: int
    num_clients: int
    use_tensorboard: bool
    num_rounds: int
    use_wandb: bool
    save_csv: bool
    checkpoint_every: int = field(default=10)
    out_prefix: str = field(default='')
    # plot_every: int = field(default=10)
    mode: str = field(default='federated')
    # flower: Optional[FlowerConfig]
    flwr_resources: dict = field(default_factory=default_resources)


    def __post_init__(self):
        assert self.mode in ['federated', 'standalone', 'centralized', 'flower'], f'Unknown simulator mode: {self.mode}'
        assert (self.use_tensorboard or self.use_wandb or self.save_csv), f'Select any one logging method atleast to avoid losing results'


########## Client Configurations ##########

@dataclass
class MetricConfig:
    eval_metrics: list
    # fairness_metrics: list
    log_to_file: bool = False
    file_prefix: str = field(default='')
    cwd: Optional[str] = field(default=os.getcwd())

@dataclass
class ClientConfig:
    epochs: int = field()
    start_epoch: int = field()
    device: str = field()
    batch_size: int = field()
    optimizer: dict = field()
    criterion: dict = field()
    lr: float = field()         # Client LR is optional
    lr_scheduler: Optional[dict] = field()
    lr_decay: Optional[float] = field()
    metric_cfg: MetricConfig = field()
    shuffle: bool = field(default=False)
    # eval_metrics: list = field(default_factory=list)
    # file_prefix: str = field(default='')
    
    def __post_init__(self):
        # if self.device =='cuda':
        
        #     assert cuda.is_available(), 'Please check if your GPU is available !' 
        assert self.batch_size >= 1
        if self.device == 'auto':
            if cuda.is_available():
                if cuda.device_count() > 1:
                    self.device = f'cuda:{get_free_gpu()}'
                else:
                    self.device = 'cuda'

            elif mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            logger.info(f'Auto Configured device to: {self.device}')
        if self.lr_scheduler:
            arg_check(self.lr_scheduler)

        arg_check(self.optimizer)

# Configs used by the server 
@dataclass
class TrainConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    optimizer: dict = field()
    criterion: dict = field()
    lr: float = field()         # Client LR is optional
    lr_scheduler: Optional[dict] = field()
    lr_decay: Optional[float] = field()
    shuffle: bool = field(default=False)
    eval_metrics: list = field(default_factory=list)
    
    def __post_init__(self):
        # if self.device =='cuda':
        #     assert cuda.is_available(), 'Please check if your GPU is available !' 
        assert self.batch_size >= 1
        if self.device == 'auto':
            if cuda.is_available():
                if cuda.device_count() > 1:
                    self.device = f'cuda:{get_free_gpu()}'
                else:
                    self.device = 'cuda'

            elif mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            logger.info(f'Auto Configured device to: {self.device}')

        if self.lr_scheduler:
            arg_check(self.lr_scheduler)

        arg_check(self.optimizer)


def default_seed():
    return [1,2,3]

@dataclass
class FedstdevClientConfig(ClientConfig):
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
    eval_batch_size: int = field(default=64)
    sampling_fraction: float = field(default=1.0)
    rounds: int = 1
    # multiprocessing: bool = False

    def __post_init__(self):
        assert self.sampling_fraction == Range(0.0, 1.0), f'Invalid value {self.sampling_fraction} for sampling fraction'
        assert self.eval_fraction == Range(0., 1.0)
        assert self.eval_type == 'both' # Remove later

# TODO: Isolate strategy configuration from server configuration to avoid duplication
@dataclass
class StrategyConfig:
    train_fraction: float
    eval_fraction: float

@dataclass
class CGSVConfig(ServerConfig):
    beta: float = 1.5
    alpha: float = 0.95
    gamma: float = 0.5
    delta_normalize: bool = False

    def __post_init__(self):
        super().__post_init__()
        

@dataclass
class FedavgConfig(ServerConfig):
    momentum: Optional[float] = float('nan')
    update_rule: str = field(default='param_average')
    delta_normalize: bool = False
    gamma: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # assert self.momentum >= 0.0
        assert self.update_rule in ['param_average', 'gradient_average']
        if self.update_rule == 'param_average' and self.delta_normalize:
            logger.warn("Delta normalize flag will be ignored in parameter averaging mode")


@dataclass
class FedstdevServerConfig(ServerConfig):
    alpha: float = 0.95
    gamma: float = 0.5
    betas: list = field(default_factory=list)
    weighting_strategy: str = field(default='tanh')
    delta_normalize: bool = False
    
    def __post_init__(self):
        # super().__post_init__()
        assert self.weighting_strategy in ['tanh', 'min_max', 'tanh_sigma_by_mu'], 'Incorrect weight scaling type'
        

@dataclass
class ServerSchema:
    _target_: str
    _partial_: bool 
    cfg: ServerConfig
    train_cfg: ClientConfig

@dataclass
class StrategySchema:
    _target_: str
    # _partial_: bool
    cfg: StrategyConfig

########## Other configurataions ##########

@dataclass
class TransformsConfig:
    resize: Optional[dict] = field(default_factory=dict)
    normalize: Optional[dict] = field(default_factory=dict)
    train_cfg: Optional[list] = field(default_factory=list)

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
class SplitConfig:
    split_type: str
    noise: Optional[NoiseConfig]
    num_splits: int  # should be equal to num_clients
    # Train test split ratio within the client
    test_fraction: float 
    def __post_init__(self):
        assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        known_splits =  ['one_noisy_client', 'one_label_flipped_client', 'iid', 'unbalanced', 'one_imbalanced_client' ]
        assert self.split_type in known_splits, f'Invalid split type: {self.split_type}'


@dataclass
class DatasetConfig:
    name: str
    data_path: str
    transforms: Optional[TransformsConfig]
    split_conf: SplitConfig
    subsample_fraction: float = 0.0  # subsample the dataset with the given fraction
    subsample: bool = False


    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        self.data_path = to_absolute_path(self.data_path)

@dataclass
class ModelSpecConfig:
    _target_: str
    _partial_: bool
    hidden_size: Optional[int]
    num_classes: Optional[int]
    in_channels: Optional[int]

@dataclass
class DatasetModelSpec:
    num_classes: int
    in_channels: int

@dataclass
class ModelConfig:
    name: str
    init_type: str
    init_gain: float
    dropout: float
    model_spec: ModelSpecConfig



########## Master Configurations ##########
@dataclass
class Config():
    mode: str = field()
    desc: str = field()
    simulator: SimConfig = field()
    server: ServerSchema = field()
    strategy: StrategySchema = field()
    client: ClientSchema = field()
    dataset: DatasetConfig = field()
    model: ModelConfig = field()
    log_conf: list = field(default_factory=list)

    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):
        # if self.dataset.use_model_tokenizer or self.dataset.use_pt_model:
        #     assert self.model.name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'

        # Set visible GPUs
        #TODO: MAke the gpu configurable
        gpu_ids = get_free_gpus()
        # logger.info('Selected GPUs:')
        logger.info('Selected GPUs:'+",".join(map(str, gpu_ids)) )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        flat_cfg = json_normalize(asdict(self))
        if not all(arg in flat_cfg for arg in self.log_conf):
            raise(KeyError(f'Recheck the keys set in log_conf: {self.log_conf}'))
        if self.mode == 'debug':
            set_debug_mode(self)



def set_debug_mode(cfg: Config):
    '''Debug mode overrides to the configuration object'''
    logger.root.setLevel(logging.DEBUG)
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = True

    logger.debug(f'[Debug Override] Setting use_wandb to: {cfg.simulator.use_wandb}')
    cfg.simulator.num_rounds = 3
    logger.debug(f'[Debug Override] Setting rounds to: {cfg.simulator.num_rounds}')
    cfg.client.cfg.epochs = 1
    logger.debug(f'[Debug Override] Setting epochs to: {cfg.client.cfg.epochs}')

    # cfg.simulator.num_clients = 3
    # cfg.dataset.num_clients = 3
    logger.debug(f'[Debug Override] Setting num clients to: {cfg.simulator.num_clients}')
    
    cfg.dataset.subsample = True
    cfg.dataset.subsample_fraction = 0.05


def register_configs():
    # Register a new configuration scheme to be validated against from the config file
    cs = ConfigStore.instance()
    cs.store(name='base_config', node=Config)
    cs.store(group='client', name='client_schema', node=ClientSchema)
    cs.store(group='client/cfg', name='base_client', node=ClientConfig)
    cs.store(group='server', name='base_server', node=ServerSchema)
    cs.store(group='server/cfg', name='base_cgsv', node=CGSVConfig)
    cs.store(group='strategy', name='strategy_schema', node=StrategySchema)
    cs.store(group='strategy/cfg', name='base_strategy', node=StrategyConfig)
    cs.store(group='server/cfg', name='base_fedavg', node=FedavgConfig)
    cs.store(group='server/cfg', name='fedstdev_server', node=FedstdevServerConfig)
    cs.store(group='client/cfg', name='fedstdev_client', node=FedstdevClientConfig)