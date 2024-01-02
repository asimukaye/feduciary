from collections import OrderedDict
from abc import ABC, abstractmethod
import functools
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
import logging
from src.metrics.metricmanager import MetricManager
from src.common.utils import log_tqdm
from src.config import ClientConfig
from src.results.resultmanager import ResultManager

logger = logging.getLogger(__name__)




class ABCClient(ABC):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self,
                 cfg: ClientConfig,
                 client_id: str,
                 dataset: tuple,
                 model: Module,
                 res_man: ResultManager = None): 
        
        self._identifier = client_id 
        # self._identifier: str = f'{id_seed:04}' # potential to convert to hash
        self._model = model
        self.res_man = res_man

        self._init_state_dict: dict = model.state_dict()

        # self._round = 0
        # self._epoch = 0
        # self._start_epoch = 0
        # self._is_resumed = False

        #NOTE: IMPORTANT: Make sure to deepcopy the config in every child class
        self.cfg = deepcopy(cfg)
        self.training_set = dataset[0]
        self.test_set = dataset[1]

        
        self.metric_mngr = MetricManager(self.cfg.eval_metrics, self._round, actor=self._identifier)
        self.optim_partial: functools.partial = self.cfg.optimizer
        self.criterion = self.cfg.criterion

        self.train_loader = self._create_dataloader(self.training_set, shuffle=cfg.shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)
        self._optimizer: Optimizer = self.optim_partial(self._model.parameters())

        # self._debug_param: Tensor = None

    @property
    def id(self)->str:
        return self._identifier

    @property
    def model(self)-> Module:
        return self._model
    
    # @model.setter
    def set_model(self, model: Module):
        self._model = model
    
    def set_lr(self, lr:float) -> None:
        self.cfg.lr = lr

    # @property
    # def _round(self)->int:
    #     return self._round
    # @_round.setter
    # def _round(self, value: int):
    #     self._round = value
    
    @property
    def epoch(self)->int:
        return self._epoch
    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value
    
    def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
        if self.cfg.batch_size == 0 :
            self.cfg.batch_size = len(self.training_set)
        return DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, shuffle=shuffle)
    

    @abstractmethod
    def reset_model(self) -> None:
        '''Define how to reset the model'''
        

    @abstractmethod
    def download(self, round:int, model_dict: dict[str, Parameter]):
        '''Download the model from the server'''
        pass

    @abstractmethod
    def upload(self) -> OrderedDict:
        '''Upload the model back to the server'''
        pass
 
    
    @abstractmethod
    def train(self, return_model=False):
        '''How the model should train'''

    @abstractmethod
    def evaluate(self):
        '''How to evaluate the model'''
        pass

    @abstractmethod
    def save_checkpoint(self, epoch=0):
        '''Define how to save a checkpoint'''
        pass
        
    @abstractmethod
    def load_checkpoint(self, ckpt_path:str):
        '''Define how to load a checkpoint'''


    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id:03} >'
    

