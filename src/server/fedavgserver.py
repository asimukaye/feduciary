
import torch
from torch import Tensor
from typing import Iterator, Tuple
from torch.nn import Parameter
import logging
from .baseserver import BaseServer, BaseOptimizer
from src.config import FedavgConfig
# from hydra.utils import instantiate
from torch.optim.lr_scheduler import LRScheduler
from src.results.resultmanager import ClientResult
logger = logging.getLogger(__name__)

# FIXME: rewrite more efficiently
class FedavgOptimizer(BaseOptimizer):

    def __init__(self, params:Tensor, **kwargs):
        self.lr = kwargs.get('lr')
        self.momentum = kwargs.get('momentum', 0.)
        defaults = dict(lr=self.lr, momentum=self.momentum)
        super(FedavgOptimizer, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        # param groups is list of params. Initialized in the torch optimizer class 
        for group in self.param_groups:
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data
                if beta > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * grad
                    delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator:Iterator[Tuple[str, Parameter]]):



        for group in self.param_groups:

            for server_param, (name, local_param) in zip(group['params'], local_param_iterator):

                if server_param.grad is None: # NOTE: grad is used as buffer to accumulate local updates!
                    server_param.grad = server_param.data.sub(local_param.data).mul(mixing_coefficient)
                else:
                    server_param.grad.add_(server_param.data.sub(local_param.data).mul(mixing_coefficient))
        
        # for name, param in self.model.named_parameters():
        #     ic(name, param.requires_grad)
class FedavgServer(BaseServer):
    name:str = 'FedAvgServer'
    def __init__(self, cfg:FedavgConfig, *args, **kwargs):

        super(FedavgServer, self).__init__(cfg, *args, **kwargs)

        # round indicator
        # print(kwargs)
        self.round = 0
        self.cfg = cfg
        
        self.server_optimizer = FedavgOptimizer(self.model.parameters(), lr=self.client_cfg.lr, momentum=self.cfg.momentum)

        # lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)
 
        
    def _aggregate(self, ids, train_results:ClientResult):
        updated_sizes = train_results.sizes
        # Calls client upload and server accumulate
        logger.info(f'[{self.name}] [Round: {self.round:03}] Aggregate updated signals!')

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(coefficient / sum(updated_sizes.values())) for identifier, coefficient in updated_sizes.items()}
        
        # accumulate weights
        for id in ids:
            locally_updated_weights_iterator = self.clients[id].upload()
            # Accumulate weights
            self.server_optimizer.accumulate(coefficients[id], locally_updated_weights_iterator)

        logger.info(f'[{self.name}] [Round: {self.round:03}] ...successfully aggregated into a new global model!')

    # def update(self):
    #     """Update the global model through federated learning.
    #     """
    #     # randomly select clients
    #     selected_ids = self._sample_random_clients()
    #     # broadcast the current model at the server to selected clients
    #     self._broadcast_models(selected_ids)

    #     # request update to selected clients
    #     train_results = self._update_request(selected_ids)
    #     # request evaluation to selected clients
    #     eval_result = self._eval_request(selected_ids)
    #     self.result_manager.log_client_eval_pre_result(eval_result)

    #     # receive updates and aggregate into a new weights
    #     self.server_optimizer.zero_grad() # empty out buffer
    #     self._aggregate(selected_ids, train_results) # aggregate local updates
        
    #     self.server_optimizer.step() # update global model with the aggregated update
    #     self.lr_scheduler.step() # update learning rate

    #     # remove model copy in clients
    #     self.reset_client_models(selected_ids)

    #     return selected_ids
