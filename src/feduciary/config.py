# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, asdict
from typing import Optional
import typing as t
import os
import subprocess
from io import StringIO
from abc import ABC
import pandas as pd
from hydra.core.config_store import ConfigStore
from torch import cuda
from hydra.utils import to_absolute_path, get_object
from feduciary.common.utils import Range
from torch import cuda
from torch.backends import mps
from pandas import json_normalize
import logging
import inspect
logger = logging.getLogger(__name__)

def arg_check(args: dict, fn: str|None = None): # type: ignore
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
    logger.debug('Returning GPU:{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free'])) # type: ignore
    return idx

########## Simulator Configurations ##########

# @dataclass
# class FlowerConfig:
#     client_resources: dict
def default_resources():
    return {"num_cpus":1, "num_gpus":0.0}


# TODO: Isolate result manager configs from this
# TODO: Develop a client and strategy compatibility checker
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
    eval_every: int = field(default=1)
    eval_type: str = field(default='both')
    mode: str = field(default='federated')
    # flower: Optional[FlowerConfig]
    flwr_resources: dict = field(default_factory=default_resources)


    def __post_init__(self):
        assert self.mode in ['federated', 'standalone', 'centralized', 'flower'], f'Unknown simulator mode: {self.mode}'
        assert (self.use_tensorboard or self.use_wandb or self.save_csv), f'Select any one logging method atleast to avoid losing results'


########## Training Configurations ##########
@dataclass
class MetricConfig:
    eval_metrics: list
    # fairness_metrics: list
    log_to_file: bool = False
    file_prefix: str = field(default='')
    cwd: Optional[str] = field(default=None)
    def __post_init__(self):
        self.cwd = os.getcwd() if self.cwd is None else self.cwd

########## Client Configurations ##########

@dataclass
class TrainConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    eval_batch_size: int = field()
    optimizer: dict = field()
    criterion: dict = field()
    lr: float = field()         # Client LR is optional
    lr_scheduler: Optional[dict] = field()
    lr_decay: Optional[float] = field()
    metric_cfg: MetricConfig = field()

    def __post_init__(self):
        assert self.batch_size >= 1
        if self.device == 'auto':
            if cuda.is_available():
                        # Set visible GPUs
                #TODO: MAke the gpu configurable
                gpu_ids = get_free_gpus()
                # logger.info('Selected GPUs:')
                logger.info('Selected GPUs:'+",".join(map(str, gpu_ids)) )
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

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


@dataclass
class ClientConfig:
    start_epoch: int = field(default=0)
    n_iters: int = field(init=False, default=-1) # TO be initialized after dataset is loaded
    data_shuffle: bool = field(default=False)

def default_seed():
    return [1,2,3]

@dataclass
class FedstdevClientConfig(ClientConfig):
    seeds: list[int] = field(default_factory=default_seed)
    client_ids: list[str] = field(default_factory=list)

@dataclass
class FedgradstdClientConfig(ClientConfig):
    seeds: list[int] = field(default_factory=default_seed)
    client_ids: list[str] = field(default_factory=list)
    abs_before_mean: bool = field(default=False)
    # def __post_init__(self):
    #     super().__post_init__()

@dataclass
class ClientSchema:
    _target_: str
    _partial_: bool
    cfg: ClientConfig
    train_cfg: TrainConfig

########## Server Configurations ##########
@dataclass
class ServerConfig:
    eval_type: str  = field(default='both')
    eval_every: int  = field(default=1)


########### Strategy Configurations ##########
@dataclass
class StrategyConfig:
    train_fraction: float
    eval_fraction: float
    def __post_init__(self):
        assert self.train_fraction == Range(0.0, 1.0), f'Invalid value {self.train_fraction} for sampling fraction'
        assert self.eval_fraction == Range(0., 1.0)

@dataclass
class FedOptConfig(StrategyConfig):
    alpha: float = 0.95
    gamma: float = 0.5
    delta_normalize: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.delta_normalize:
            assert self.gamma > 0.0, 'Gamma should be greater than 0 for delta normalization'
        assert self.alpha == Range(0.0, 1.0), f'Invalid value {self.alpha} for alpha'

@dataclass
class FedstdevConfig(StrategyConfig):
    '''Config schema for Fedstdev strategy config'''
    weighting_strategy: str
    betas: list[float]
    beta_0: float
    alpha: float
    num_clients: int

@dataclass
class CGSVConfig(StrategyConfig):
    num_clients: int
    beta: float = 1.5
    alpha: float = 0.95
    gamma: float = 0.15
    delta_normalize: bool = False
    sparsify_gradients: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.delta_normalize:
            assert self.gamma > 0.0, 'Gamma should be greater than 0 for delta normalization'
        assert self.alpha == Range(0.0, 1.0), f'Invalid value {self.alpha} for alpha'
        assert self.beta >= 1.0, f'Invalid value {self.beta} for beta'

# @dataclass
# class FedavgConfig(StrategyConfig):
#     momentum: Optional[float] = float('nan')
#     update_rule: str = field(default='param_average')
#     delta_normalize: bool = False
#     gamma: float = 1.0

#     def __post_init__(self):
#         super().__post_init__()
#         # assert self.momentum >= 0.0
#         assert self.update_rule in ['param_average', 'gradient_average']
#         if self.update_rule == 'param_average' and self.delta_normalize:
#             logger.warn("Delta normalize flag will be ignored in parameter averaging mode")

@dataclass
class FedavgManualConfig(StrategyConfig):
    weights : list[float]
    num_clients: int

    def __post_init__(self):
        super().__post_init__()
        assert len(self.weights) == self.num_clients, 'Number of weights should be equal to number of clients'
        # Auto normalize the weights
        self.weights = [w/sum(self.weights) for w in self.weights]
        

@dataclass
class ServerSchema:
    _target_: str
    _partial_: bool 
    cfg: ServerConfig
    train_cfg: TrainConfig

@dataclass
class StrategySchema:
    _target_: str
    # _partial_: bool
    cfg: StrategyConfig

########## Dataset configurataions ##########

#TODO: Add support for custom transforms 
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
    mu: t.Any
    sigma: t.Any
    flip_percent: t.Any
    
@dataclass
class SplitConfig:
    split_type: str
    noise: NoiseConfig
    num_splits: int  # should be equal to num_clients
    num_patho_clients: int
    num_class_per_client: int
    dirichlet_alpha: float
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    test_fractions: list[float] = field(init=False, default_factory=list) 
    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        known_splits =  {'one_noisy_client',
                         'n_noisy_clients',
                         'n_distinct_noisy_clients',
                         'n_distinct_label_flipped_clients',
                         'one_label_flipped_client',
                         'n_label_flipped_clients',
                         'iid', 'unbalanced',
                         'one_imbalanced_client',
                          'patho', 'dirichlet' }
        if self.split_type in {'one_noisy_client',
                               'n_noisy_clients',
                               'one_label_flipped_client',
                               'n_label_flipped_clients'}:
            assert self.noise, 'Noise config should be provided for noisy client or label flipped client'
        if self.split_type == 'patho':
            assert self.num_class_per_client, 'Number of pathological splits should be provided'
        if self.split_type == 'dirichlet':
            assert self.dirichlet_alpha, 'Dirichlet alpha should be provided for dirichlet split'

        assert self.split_type in known_splits, f'Invalid split type: {self.split_type}'
        assert self.num_patho_clients <= self.num_splits, 'Number of pathological splits should be less than or equal to number of splits'


@dataclass
class DatasetConfig:
    name: str
    data_path: str
    dataset_family: str
    transforms: Optional[TransformsConfig]
    test_fraction: Optional[float]
    seed: Optional[int]
    federated: bool
    split_conf: SplitConfig
    subsample: bool = False
    subsample_fraction: float = 0.0  # subsample the dataset with the given fraction


    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        self.data_path = to_absolute_path(self.data_path)
        if self.federated == False:
            assert self.split_conf.num_splits == 1, 'Non-federated datasets should have only one split'

########## Model Configurations ##########
@dataclass
class ModelConfig:
    _target_: str
    hidden_size: int

@dataclass
class ModelConfigGN(ModelConfig):
    num_groups: int

@dataclass
class DatasetModelSpec:
    num_classes: int
    in_channels: int

@dataclass
class ModelInitConfig:
    init_type: str
    init_gain: float


########## Master Configurations ##########
@dataclass
class Config():
    mode: str = field()
    desc: str = field()
    simulator: SimConfig = field()
    server: ServerSchema = field()
    strategy: StrategySchema = field()
    client: ClientSchema = field()
    train_cfg: TrainConfig = field()

    dataset: DatasetConfig = field()
    model: ModelConfig = field()
    model_init: ModelInitConfig = field()
    log_conf: list = field(default_factory=list)

    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):
        # if self.dataset.use_model_tokenizer or self.dataset.use_pt_model:
        #     assert self.model.name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'
        if self.simulator.mode == 'centralized':
            self.dataset.federated = False
            logger.info('Setting federated cfg in dataset cfg to False')
        else:
            assert self.dataset.split_conf.num_splits == self.simulator.num_clients, 'Number of clients in dataset and simulator should be equal'

        # flat_cfg = json_normalize(asdict(self))
        # if not all(arg in flat_cfg for arg in self.log_conf):
        #     raise(KeyError(f'Recheck the keys set in log_conf: {self.log_conf}'))
        if self.mode == 'debug':
            set_debug_mode(self)


########## Debug Configurations ##########
def set_debug_mode(cfg: Config):
    '''Debug mode overrides to the configuration object'''
    logger.root.setLevel(logging.DEBUG)
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = True

    logger.debug(f'[Debug Override] Setting use_wandb to: {cfg.simulator.use_wandb}')
    cfg.simulator.num_rounds = 2
    logger.debug(f'[Debug Override] Setting rounds to: {cfg.simulator.num_rounds}')
    cfg.client.train_cfg.epochs = 1
    logger.debug(f'[Debug Override] Setting epochs to: {cfg.client.train_cfg.epochs}')

    cfg.simulator.num_clients = 3
    cfg.dataset.split_conf.num_splits = 3
    if cfg.dataset.split_conf.split_type in ['n_label_flipped_clients', 'n_noisy_clients', 'n_distinct_noisy_clients', 'n_distinct_label_flipped_clients']:
        cfg.dataset.split_conf.num_patho_clients = 2
    if hasattr(cfg.strategy.cfg, 'num_clients'):
        cfg.strategy.cfg.num_clients = 3 # type: ignore
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
    cs.store(group='server', name='base_flower_server', node=ServerSchema)

    cs.store(group='train_cfg', name='base_train', node=TrainConfig)


    cs.store(group='strategy/cfg', name='base_cgsv', node=CGSVConfig)
    cs.store(group='strategy', name='strategy_schema', node=StrategySchema)
    cs.store(group='strategy/cfg', name='base_strategy', node=StrategyConfig)
    cs.store(group='strategy/cfg', name='fedavgmanual', node=FedavgManualConfig)
    cs.store(group='strategy/cfg', name='fedopt', node=FedOptConfig)
    cs.store(group='strategy/cfg', name='cgsv', node=CGSVConfig)
    cs.store(group='model', name='resnet18gn', node=ModelConfigGN)
    cs.store(group='model', name='resnet34gn', node=ModelConfigGN)
    cs.store(group='model', name='resnet10gn', node=ModelConfigGN)
    # cs.store(group='model', name='resnet18gn', node=ModelConfigGN)


    cs.store(group='strategy/cfg', name='fedstdev_strategy', node=FedstdevConfig)
    # cs.store(group='server/cfg', name='base_fedavg', node=FedavgConfig)
    # cs.store(group='server/cfg', name='fedstdev_server', node=FedstdevServerConfig)
    cs.store(group='client/cfg', name='fedstdev_client', node=FedstdevClientConfig)
    cs.store(group='client/cfg', name='fedgradstd_client', node=FedgradstdClientConfig)