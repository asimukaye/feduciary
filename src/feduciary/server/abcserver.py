from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple
import logging
import torch
import random
import gc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, ParameterDict, Parameter
from torch import Tensor

from torch.optim.lr_scheduler import LRScheduler

from feduciary.client.baseclient import BaseClient, model_eval_helper
from feduciary.metrics.metricmanager import MetricManager
from feduciary.common.utils  import log_tqdm, log_instance
# from feduciary.common.typing import ClientParams_t

from feduciary.results.resultmanager import ResultManager
from feduciary.strategy.abcstrategy import ABCStrategy

# from concurrent.futures import ThreadPoolExecutor, as_completed

from feduciary.config import *

# TODO: Long term todo: the server should 
#  eventually be tied directly to the server algorithm
logger = logging.getLogger(__name__)
    
    
class ABCServer(ABC):
    """Central server orchestrating the whole process of federated learning.
    """
    name: str = 'ABCServer'
    _round: int = 0

    # NOTE: It is must to redefine the init function for child classes with a call to super.__init__()
    @abstractmethod
    def __init__(self,
                 clients: dict[str, BaseClient],
                 model: Module,
                 cfg: ServerConfig,
                 strategy: ABCStrategy,
                 train_cfg: ClientConfig,
                 dataset: Dataset,
                 result_manager: ResultManager):
        pass

    
    @abstractmethod
    def _broadcast_models(self, ids: list[str]):
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """
        pass


    @abstractmethod
    def server_eval(self):
        """Evaluate the global model located at the server.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, ckpt_path):
        pass
    
    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass
   
    @abstractmethod
    def update(self):
        pass

    
    # Every server needs to implement this function uniquely
    @abstractmethod
    def local_eval(self, client_ids: list[str], *args):
        # receive updates and aggregate into a new weights
        # Below is a template for how aggregation might work

        pass
