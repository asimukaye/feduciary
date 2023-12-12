from collections import OrderedDict

from torch.nn import Module
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

    def __init__(self, model: Module, client_lr: float, cfg: FedavgConfig) -> None:
        super().__init__(model, client_lr, cfg)
        self.cfg = cfg


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
                # client delta =  client param(w_k+1,i) - server param (w_k)
                client_delta = client_param[key].data.sub(server_param.data)
                
                # if self.gradient_normalize:
                #     norm = client_delta.norm() 
                #     if norm == 0:
                #         logger.warning(f"CLIENT [{cid}]: Got a zero norm")
                #         client_delta_norm = client_delta.mul(self.gamma)
                #     else:
                #         client_delta_norm = client_delta.div(norm).mul(self.gamma)
                if self._server_deltas[key] is None:
                    self._server_deltas[key] = self._client_weights[cid] * client_delta
                else:
                    self._server_deltas[key].add_(self._client_weights[cid] * client_delta)

        for key, delta in self._server_deltas.items():
            self._server_params[key].data.add_(delta)


    def param_average_update(self) -> None:
         for key in self._server_params.keys():
            temp_parameter = None

            for cid, client_param in self._client_params.items():
                if temp_parameter is None:
                    temp_parameter = self._client_weights[cid] * client_param[key].data
                else:
                    temp_parameter.data.add_(self._client_weights[cid] * client_param[key].data)
            
            self._server_params[key].data = temp_parameter.data


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
        
        self.server_optimizer = FedavgOptimizer(self.model, self.client_cfg.lr, self.cfg)

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
            self.result_manager.log_parameters(client_params, 'pre_agg', cid, verbose=False)
            # locally_updated_weights_iterator = self.clients[cid].upload()
            # # Accumulate weights
            # self.server_optimizer.accumulate(coefficients[cid], locally_updated_weights_iterator)

        self.server_optimizer.aggregate(train_results.sizes)
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # Full parameter debugging
        self.result_manager.log_general_metric(self.server_optimizer._client_weights, 'client_weights', 'post_agg', 'server')
        self.result_manager.log_parameters(self.model.state_dict(), 'post_agg', 'server')
        # for cid in client_ids:
        #     self.result_manager.log_parameters(client_params, 'post_agg', cid)

        logger.info(f'[{self.name}] [Round: {self.round:03}] successfully aggregated into a new global model!')