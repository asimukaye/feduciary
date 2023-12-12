from collections import OrderedDict
import torch
from torch import Tensor
from typing import Iterator, Tuple, Iterable
import logging
from .baseserver import BaseServer, BaseStrategy
from src.config import FedavgConfig, ServerConfig
# from hydra.utils import instantiate
from torch.optim.lr_scheduler import LRScheduler
from src.results.resultmanager import ClientResult
logger = logging.getLogger(__name__)
from torch.optim import SGD

class FedavgOptimizer(BaseStrategy):

    def __init__(self, params: OrderedDict, client_lr: float, cfg: FedavgConfig) -> None:

        super().__init__(params, client_lr, cfg)
        self.cfg = cfg

    # def step(self):
    #     # param groups is list of params. Initialized in the torch optimizer class 

    #     for group in self.param_groups:
    #         beta = group['momentum']
    #         for param in group['params']:
    #             if param.grad is None:
    #                 continue
    #             delta = param.grad.data
    #             if beta > 0.:
    #                 if 'momentum_buffer' not in self.state[param]:
    #                     self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
    #                 self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * grad
    #                 delta = self.state[param]['momentum_buffer']
    #             param.data.sub_(delta)

    def param_update_rule(self) -> None:
        if self.cfg.update_rule == 'param_average':
            self.param_average_update()
        elif self.cfg.update_rule == 'gradient_average':
            self.gradient_average_update()
        else:
            raise ValueError('Unkown param update rule')
        
    
    def gradient_average_update(self) -> None:
        for key, server_param in self._server_params.items():
            for cid, client_param in self._client_params.items():
                # Using FedNova Notation of delta (Δ) as (-grad ∇)
                client_delta = -1 * (client_param[key] - server_param)
                self._server_deltas[key] += self._client_weights[cid] * client_delta

        for key, delta in self._server_deltas.items():
            self._server_params[key].data.add_(delta)


    def param_average_update(self) -> None:
         for key in self._server_params.keys():
            for cid, client_param in self._client_params.items():
                self._server_params[key].data += self._client_weights[cid] * client_param[key].data

    # def accumulate(self, mixing_coefficient, local_param_iterator: Iterator[Tuple[str, Parameter]]):

    def aggregate(self, client_data_sizes: dict[str, int]):
        # calculate client weights according to sample sizes
        self._client_weights = {}
        for cid, data_size in client_data_sizes.items():
            self._client_weights[cid] = float(data_size / sum(client_data_sizes.values())) 


class FedavgServer(BaseServer):
    name:str = 'FedAvgServer'
    def __init__(self, cfg: FedavgConfig, *args, **kwargs):

        super(FedavgServer, self).__init__(cfg, *args, **kwargs)

        self.round = 0
        self.cfg = cfg
        
        self.server_optimizer = FedavgOptimizer(self.model.state_dict(), self.client_cfg.lr, self.cfg)

        # Global lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)
 
        
    def _run_strategy(self, client_ids: list[str], train_results: ClientResult):
        # updated_sizes = train_results.sizes
        # Calls client upload and server accumulate
        logger.debug(f'[{self.name}] [Round: {self.round:03}] Aggregate updated signals!')
        self.result_manager.log_parameters(self.model.state_dict(), 'pre_agg', 'server')


        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer
        # accumulate weights
        for cid in client_ids:
            client_params = self.clients[cid].upload()
            self.server_optimizer.set_client_params(cid, client_params)
            self.result_manager.log_parameters(client_params, 'pre_agg', cid)
            # locally_updated_weights_iterator = self.clients[cid].upload()
            # # Accumulate weights
            # self.server_optimizer.accumulate(coefficients[cid], locally_updated_weights_iterator)

        self.server_optimizer.aggregate(train_results.sizes)
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # Full parameter debugging
        self.result_manager.log_parameters(self.model.state_dict(), 'post_agg', 'server')
        for cid in client_ids:
            self.result_manager.log_parameters(client_params, 'post_agg', cid)

        logger.info(f'[{self.name}] [Round: {self.round:03}] successfully aggregated into a new global model!')