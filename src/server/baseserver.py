from abc import ABC, abstractmethod
from typing import List
import logging
import json
import torch
import random
import gc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate
from src.client.baseclient import BaseClient, model_eval_helper

from src.utils  import TqdmToLogger, log_instance
from src.metrics.metricmanager import MetricManager, ResultManager, ClientResult

from importlib import import_module
# from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import *

# TODO: Long term todo: the server should probably eventually be tied directly to server algorithm
logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    # FIXME: LR is a required parameter as per current implementation
    @abstractmethod
    def step(self, closure=None):
        raise NotImplementedError
     
    @abstractmethod
    def accumulate(self, **kwargs):
        raise NotImplementedError

class BaseServer(ABC):
    """Centeral server orchestrating the whole process of federated learning.
    """
    name:str = 'BaseServer'
    def __init__(self, cfg:ServerConfig, client_cfg: ClientConfig, model:Module, dataset:Dataset, clients:list[BaseClient], writer:SummaryWriter):
        self.round = 0
        self.model = model
        self.clients: dict[str, BaseClient] = clients
        self.num_clients:int = len(self.clients)
        self.writer = writer
        self.cfg = cfg
        self.client_cfg = client_cfg

        self.result_manager = ResultManager(logger=logger, writer=writer)
        # global holdout set
        # if self.cfg.eval_type != 'local':
        self.server_dataset = dataset
        self.lr_scheduler:LRScheduler 
        # print(self.client_cfg.lr_scheduler)
        # print(type(self.client_cfg.lr_scheduler))
    
    # @log_instance(attrs=['round'], m_logger=logger)
    def _broadcast_models(self, ids:list[str]):
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """
        def __broadcast_model(client: BaseClient):
            client.download(self.round, self.model)
        
        self.model.to('cpu')

        for identifier in TqdmToLogger(
            ids, logger=logger, desc=f'[{self.name}] [Round: {self.round:03}] ...broadcasting server model... ',total=len(ids)):
            __broadcast_model(self.clients[identifier])
      


    @log_instance(attrs=['round'], m_logger=logger)
    def _sample_random_clients(self):
        # NOTE: Update does not use the logic of C+ 0 meaning all clients
        
        # logger.info(f'[{self.name}] [Round: {self.round:03}] Sample clients!')
        
        # Update - randomly select max(floor(C * K), 1) clients
        num_sampled_clients = max(int(self.cfg.sampling_fraction * self.num_clients), 1)
        sampled_client_ids = sorted(random.sample([cid for cid in self.clients.keys()], num_sampled_clients))

        logger.info(f'[{self.name}] [Round: {self.round:03}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    
    def _sample_selected_clients(self, exclude: list[str]):
        # FIXME: Rewrite this for clarity of usage
        num_unparticipated_clients = self.num_clients- len(exclude)
        if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
            num_sampled_clients = self.num_clients
            sampled_client_ids = sorted(self.clients.keys())
        else:
            num_sampled_clients = max(int(self.cfg.eval_fraction * num_unparticipated_clients), 1)
            sampled_client_ids = sorted(random.sample([identifier for identifier in self.clients.keys() if identifier not in exclude], num_sampled_clients))
       
        logger.info(f'[{self.name}] [Round: {self.round:03}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids
    

    def _update_request(self, ids:list[str]) -> ClientResult:
        def __update_clients(client:BaseClient):
            # getter function for client update
            if self.lr_scheduler:
                client.cfg.lr = self.lr_scheduler.get_last_lr()[-1]
            update_result = client.update()
            return (client.id, update_result)
        
        results_list = []
        for idx in TqdmToLogger(ids, logger=logger, desc=f'[{self.name}] [Round: {self.round:03}] ...receiving updates... ', total=len(ids)):
            self.clients[idx].cfg.lr = self.lr_scheduler.get_last_lr()[-1]
            results_list.append(__update_clients(self.clients[idx]))
    
        results_dict = dict(results_list)
        update_result = self.result_manager.log_client_train_result(results_dict)

        logger.info(f'[{self.name}] [Round: {self.round:03}] ...completed updates of {"all" if ids is None else len(ids)} clients!')

        return update_result

    
    def _eval_request(self, ids)->dict:
        
        def __evaluate_clients(client: BaseClient):
            eval_result = client.evaluate() 
            return (client.id, eval_result)

        # if self.args._train_only: return
        results = []
        for idx in TqdmToLogger(
            ids, logger=logger, 
            desc=f'[{self.name}] [Round: {self.round:03}] ...evaluate clients... ',
            total=len(ids)
            ):
            results.append(__evaluate_clients(self.clients[idx]))

        eval_results = dict(results)
        
        logger.info(f'[{self.name}] [Round: {self.round:03}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')

        return eval_results

    def _cleanup(self, indices):
        logger.info(f'[{self.name}] [Round: {self.round:03}] Clean up!')

        for identifier in indices:
            if self.clients[identifier].model is not None:
                self.clients[identifier].reset_model()
            else:
                err = f'why clients ({identifier}) has no model? please check!'
                logger.exception(err)
                raise AssertionError(err)
        logger.info(f'[{self.name}] [Round: {self.round:03}] ...successfully cleaned up!')
        gc.collect()

    @torch.inference_mode()
    def _central_evaluate(self):

        server_loader = DataLoader(dataset=self.server_dataset, batch_size=self.client_cfg.batch_size, shuffle=False)
        # log result
        result = model_eval_helper(self.model, server_loader, self.client_cfg, 'server', self.round)
        return result



    def evaluate(self, excluded_ids):
        # FIXME: Rewrite this for clarity of usage
        """Evaluate the global model located at the server.
        """
        # randomly select all remaining clients not participated in current round
        selected_ids = self._sample_selected_clients(exclude=excluded_ids)
        self._broadcast_models(selected_ids)

        eval_results = self._eval_request(selected_ids)
        server_results = self._central_evaluate()

        self.result_manager.log_client_eval_result(eval_results)
        self.result_manager.log_server_eval_result(server_results)

        # remove model copy in clients
        self._cleanup(selected_ids)

    
    def save_checkpoint(self, epoch):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            }, f'model_ckpt_{epoch:003}.pt')

    
    def finalize(self):
        self.result_manager.finalize()
        # save checkpoint
        torch.save(self.model.state_dict(), f'final_model.pt')
        
   

    # Every server needs to implement these uniquely

    @abstractmethod
    def _aggregate(self, indices, update_sizes):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError
