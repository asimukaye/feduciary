from collections import OrderedDict
from dataclasses import dataclass, field
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
from src.config import ClientConfig, TrainConfig
from src.results.resultmanager import ResultManager
import src.common.typing as fed_t
logger = logging.getLogger(__name__)



class ABCClient(ABC):
    """Class for client object having its own (private) data and resources to train a model.
    """

    def __init__(self,
                 cfg: ClientConfig,
                 train_cfg: TrainConfig,
                 client_id: str,
                 dataset: tuple,
                 model: Module): 
        
        self._cid = client_id 
        self._model = model

        # self._init_state_dict: dict = model.state_dict()

        # #NOTE: IMPORTANT: Make sure to deepcopy the config in every child class
        self.cfg = deepcopy(cfg)
        self.train_cfg = deepcopy(train_cfg)

        self.training_set = dataset[0]
        self.test_set = dataset[1]


    # @property
    # def id(self)->str:
    #     return self._cid

    # @property
    # def model(self)-> Module:
    #     return self._model
    
    # # @model.setter
    # def set_model(self, model: Module):
    #     self._model = model
    
    # def set_lr(self, lr:float) -> None:
    #     self.train_cfg.lr = lr

    # @property
    # def _round(self)->int:
    #     return self._round
    # @_round.setter
    # def _round(self, value: int):
    #     self._round = value
    
    # @property
    # def epoch(self)->int:
    #     return self._epoch
    # @epoch.setter
    # def epoch(self, value: int):
    #     self._epoch = value
    
    # def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
    #     if self.train_cfg.batch_size == 0 :
    #         self.train_cfg.batch_size = len(self.training_set)
    #     return DataLoader(dataset=dataset, batch_size=self.train_cfg.batch_size, shuffle=shuffle)
    
    @abstractmethod
    def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
        pass
    
    @dataclass
    class ClientInProtocol:
        server_params: fed_t.ActorParams_t

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
    def unpack_train_input(self, client_ins: fed_t.ClientIns) -> ClientInProtocol:
        pass
    
    @abstractmethod
    def pack_train_result(self, result: fed_t.Result) -> fed_t.ClientResult1:
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
        return f'CLIENT < {self._cid:03} >'
    

