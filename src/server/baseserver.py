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

import torch.multiprocessing as torch_mp
from torch.multiprocessing import Queue
from logging.handlers import  QueueListener, QueueHandler
from torch.optim.lr_scheduler import LRScheduler

from src.client.baseclient import BaseClient, model_eval_helper
from src.metrics.metricmanager import MetricManager
from src.utils  import log_tqdm, log_instance, ClientParams
from src.results.resultmanager import AllResults, ResultManager, ClientResult, Result


# from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import *

# TODO: Long term todo: the server should 
#  eventually be tied directly to the server algorithm
logger = logging.getLogger(__name__)

def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(qh)

def add_logger_queue()-> Tuple[Queue, QueueListener]:
    q = Queue(1000)
    ql = QueueListener(q, logger.root.handlers[0], logger.root.handlers[1])
    ql.start()
    return q, ql

def update_client(client: BaseClient):
    # getter function for client update
    update_result, model = client.train()
    return {'id':client.id, 'result':update_result, 'model':model}

class BaseStrategy(torch.optim.Optimizer, ABC):
    # A strategy is a server strategy used to aggregate the server weights
    # It is based on the torch optimizer class to support smooth integrations of Pytorch LR schedulers
    # loss: Tensor

    def __init__(self, params: OrderedDict, client_lr: float, cfg: ServerConfig) -> None:

        # self.cfg = cfg
        # NOTE: Client LR is required to make sure correct LR scheduling over rounds
        defaults = dict(lr=client_lr)

        super().__init__(params.values(), defaults)
        assert len(self.param_groups) == 1, f'Multi param group yet to be implemented'
        self._server_params: OrderedDict[str, Parameter] = params
        self._server_deltas: OrderedDict[str, Tensor] = defaultdict()

        self._client_params: ClientParams = defaultdict(dict)
        self._client_weights: dict[str, float] = defaultdict()


    def zero_grad(self, set_to_none: bool = ...) -> None:
        for key in self._server_deltas.keys():
            self._server_deltas[key] = None
        # Maybe we need to add seting server param grads to NOne as well
        super().zero_grad(set_to_none)

    def step(self) -> None:
        
        ic(type(self.param_groups[0]['params']))
        # Let's check for the object reference
        ic(id(self.param_groups[0]['params'][0]))
        ic(id(list(self._server_params.values())[0]))

        # Apply the update rule
        self.param_update_rule()

        # Map the server param dictionary back to the optimizer classes params
        self.param_groups[0]['params'] = list(self._server_params.values())

        ic(id(self.param_groups[0]['params'][0]))
        ic(id(list(self._server_params)[0]))

        
    def set_client_params(self, cid: str, params: OrderedDict) -> None:
        self._client_params[cid] = params
    
    @abstractmethod
    def param_update_rule(self) -> None:
        '''Define how to update the server parameters from the server deltas.
        Needs to modify the server params'''

        # Example of server param update with unit learning rate, child classes need to implement this individually
        for key, delta in self._server_deltas.items():
            self._server_params[key].data.add_(delta)
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, **kwargs):
        '''Fill in client parameter aggregation logic here'''
        raise NotImplementedError
    
    

class BaseServer(ABC):
    """Centeral server orchestrating the whole process of federated learning.
    """
    name: str = 'BaseServer'
    def __init__(self, cfg: ServerConfig, client_cfg: ClientConfig, model: Module, dataset: Dataset, clients: list[BaseClient], result_manager: ResultManager):
        self.round = 0
        self.model = model
        self.clients: dict[str, BaseClient] = clients
        self.num_clients:int = len(self.clients)
        # self.writer = writer
        self.cfg = cfg
        self.client_cfg = client_cfg
        self.server_optimizer: BaseStrategy = None
        self.loss: torch.Tensor = None
        self.lr_scheduler: LRScheduler = None


        self.result_manager = result_manager
        self.metric_manager = MetricManager(eval_metrics=client_cfg.eval_metrics,round= 0, caller='server')

        # global holdout set
        # wandb.watch(self.model, log='all', log_freq=5)
        # if self.cfg.eval_type != 'local':
        self.server_dataset = dataset

    
    # @log_instance(attrs=['round'], m_logger=logger)
    def _broadcast_models(self, ids:list[str]):
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """
        def __broadcast_model(client: BaseClient):
            # NOTE: Consider setting keep vars to true later if gradients are required
            client.download(self.round, self.model.state_dict())

        # Uncomment this when adding GPU support to server
        # self.model.to('cpu')

        for idx in log_tqdm(ids, desc=f'broadcasting models: ', logger=logger):
            __broadcast_model(self.clients[idx])


    @log_instance(attrs=['round'], m_logger=logger)
    def _sample_random_clients(self)-> list[str]:
        # NOTE: Update does not use the logic of C+ 0 meaning all clients

        # Update - randomly select max(floor(C * K), 1) clients
        num_sampled_clients = max(int(self.cfg.sampling_fraction * self.num_clients), 1)
        sampled_client_ids = sorted(random.sample([cid for cid in self.clients.keys()], num_sampled_clients))

        logger.debug(f'[{self.name}] [Round: {self.round:03}] {num_sampled_clients} clients are randomly selected')
        return sampled_client_ids

    
    def _sample_selected_clients(self, exclude: list[str]) -> list[str]:
        # FIXME: Rewrite this for clarity of usage
        num_unparticipated_clients = self.num_clients - len(exclude)
        if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
            num_sampled_clients = self.num_clients
            sampled_client_ids = sorted(self.clients.keys())
        else:
            num_sampled_clients = max(int(self.cfg.eval_fraction * num_unparticipated_clients), 1)
            sampled_client_ids = sorted(random.sample([identifier for identifier in self.clients.keys() if identifier not in exclude], num_sampled_clients))
       
        logger.debug(f'[{self.name}] [Round: {self.round:03}] {num_sampled_clients} clients are selected')
        return sampled_client_ids
    

    def _update_request(self, ids:list[str]) -> ClientResult:
        def __update_client(client: BaseClient):
            # getter function for client update
            update_result = client.train()
            return {'id':client.id, 'result':update_result}
        results_list = []
  
        # Set the LRs for each client before training
        # TODO: does lr scheduling need to be done for select ids ??
        if self.lr_scheduler:
            current_lr = self.lr_scheduler.get_last_lr()[-1]
            [client.set_lr(current_lr) for client in self.clients.values()]

        logger.debug(f'[{self.name}] [Round: {self.round:03}] Beginning updates')

        # ctx = torch_mp.get_context('spawn')
        if self.cfg.multiprocessing:
            q, ql = add_logger_queue()
            with torch_mp.Pool(len(ids), worker_init, [q]) as pool:
            # with torch_mp.Pool(len(ids)) as pool:
                results_list = pool.map(update_client, [self.clients[idx]  for idx in ids])
            ql.stop()
            q.close()
            # FIXME: Changing implementation to use state dictionary rather than modules
            [self.clients[item['id']].set_model(item['model']) for item in results_list]
        else:
             for idx in ids:
                results_list.append(__update_client(self.clients[idx]))

        # results_dict = dict(results_list)
        results_dict = {item['id']: item['result'] for item in results_list}
        # update_result = self.result_manager.log_client_train_result(results_dict)
        update_result = self.result_manager.log_client_result(results_dict, key='client_eval')

        logger.info(f'[{self.name}] [Round: {self.round:03}] ...completed updates of {"all" if ids is None else len(ids)} clients.')

        return update_result

    
    def _eval_request(self, ids)->dict[str, Result]:
        
        def __evaluate_clients(client: BaseClient):
            eval_result = client.evaluate() 
            return (client.id, eval_result)

        # if self.args._train_only: return
        results = []
        for idx in log_tqdm(ids, desc='eval clients: ', logger=logger):
            results.append(__evaluate_clients(self.clients[idx]))

        eval_results = dict(results)
        
        logger.debug(f'[{self.name}] [Round: {self.round:03}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')

        return eval_results

    def reset_client_models(self, indices):
        logger.debug(f'[{self.name}] [Round: {self.round:03}] Clean up!')

        for identifier in indices:
            if self.clients[identifier].model is not None:
                self.clients[identifier].reset_model()
            else:
                err = f'Client ({identifier}) has no model.'
                logger.exception(err)
                raise AssertionError(err)
        logger.debug(f'[{self.name}] [Round: {self.round:03}] Client Models are reset')
        gc.collect()

    @torch.no_grad()
    def _central_evaluate(self):

        server_loader = DataLoader(dataset=self.server_dataset, batch_size=self.client_cfg.batch_size, shuffle=False)
        # log result
        result = model_eval_helper(self.model, server_loader, self.client_cfg, self.metric_manager, self.round)
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

        self.result_manager.log_client_result(eval_results, key='client_eval')
        self.result_manager.log_server_eval_result(server_results)

        # remove model copy in clients
        # self.reset_client_models(selected_ids)
        return selected_ids

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.server_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        self.round = checkpoint['round']
        # Find a way to avoid this result manager round bug repeatedly
        self.result_manager._round = checkpoint['round']

        # loss = checkpoint['loss']
    
    def save_checkpoint(self):
        torch.save({
            'round': self.round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict' : self.server_optimizer.state_dict(),
            }, f'server_ckpts/server_ckpt_{self.round:003}.pt')

    
    def finalize(self) -> None:
        # all_results: AllResults = self.result_manager.finalize()
        # save checkpoint
        torch.save(self.model.state_dict(), f'final_model.pt')
        # return all_results
        
   
    def update(self):
        """Update the global model through federated learning.
        """
        # randomly select clients

        # for name, param in self.model.named_parameters():
        #     ic(name, param.requires_grad)
        selected_ids = self._sample_random_clients()

        # broadcast the current model at the server to selected clients
        self._broadcast_models(selected_ids)

        # ic(self.clients[selected_ids[0]].optim_partial)
        # request update to selected clients
        train_results = self._update_request(selected_ids)
    
        # request evaluation to selected clients
        eval_result = self._eval_request(selected_ids)
        self.result_manager.log_client_result(eval_result, key='client_eval_pre')

        # self.result_manager.log_client_eval_pre_result(eval_result)

        self._run_strategy(selected_ids, train_results) # aggregate local updates


        # remove model copy in clients
        self.reset_client_models(selected_ids)

        return selected_ids

    
    # Every server needs to implement this function uniquely
    @abstractmethod
    def _run_strategy(self, client_ids: list[str], *args):
        # receive updates and aggregate into a new weights
        # Below is a template for how aggregation might work


        #### INSERT ACCUMULATION INIT HERE #####

        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer

        for cid in client_ids:
            self.server_optimizer.set_client_params(cid, self.clients[cid].upload())

        self.server_optimizer.aggregate()
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate
        raise NotImplementedError
