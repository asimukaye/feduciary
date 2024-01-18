from collections import OrderedDict
from torch import Tensor
from torch.nn import Module
from typing import Iterator, Tuple, Iterable
import logging
from .baseserver import BaseServer, BaseStrategy
from feduciary.config import FedavgConfig, ServerConfig
# from hydra.utils import instantiate
from torch.optim.lr_scheduler import LRScheduler
from feduciary.results.resultmanager import ClientResultStats
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
    
    def _delta_normalize(self, delta: Tensor, gamma: float) -> Tensor:
        '''Normalize the parameter delta update and scale to prevent potential gradient explosion'''
        norm = delta.norm() 
        if norm == 0:
            logger.warning(f"Normalize update: Got a zero norm update")
            delta_norm = delta.mul(gamma)
        else:
            delta_norm = delta.div(norm).mul(gamma)
        return delta_norm
    
    def gradient_average_update(self) -> None:
        for key, server_param in self._server_params.items():
            for cid, client_param in self._client_params.items():
                # Using FedNova Notation of delta (Δ) as (-grad ∇)
                # client delta =  client param(w_k+1,i) - server param (w_k)
                client_delta = client_param[key].data.sub(server_param.data)
                
                if self.cfg.delta_normalize:
                    client_delta = self._delta_normalize(client_delta, self.cfg.gamma)

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
    name = 'FedAvgServer'
    def __init__(self, cfg: FedavgConfig, *args, **kwargs):

        super(FedavgServer, self).__init__(cfg, *args, **kwargs)

        self._round = 0
        self.cfg = cfg
        
        self.server_optimizer = FedavgOptimizer(self.model, self.client_cfg.lr, self.cfg)

        # Global lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)
 
        
    def _run_strategy(self, client_ids: list[str], train_results: ClientResultStats):
        # updated_sizes = train_results.sizes
        # Calls client upload and server accumulate
    
    
        self.result_manager.log_parameters(self.model.state_dict(), phase='pre_agg', actor='server', verbose=False)


        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer
        # accumulate weights


        for cid in client_ids:
            client_params = self.clients[cid].upload()
            self.server_optimizer.set_client_params(cid, client_params)

            self.result_manager.log_parameters(client_params, phase='pre_agg', actor=cid, verbose=False)

        self.server_optimizer.aggregate(train_results.sizes)
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # Full parameter debugging
        self.result_manager.log_general_metric(self.server_optimizer._client_weights, metric_name='client_weights', phase='post_agg', actor='server')

        self.result_manager.log_parameters(self.model.state_dict(), phase='post_agg', actor='server', verbose=False)
        self.result_manager.log_duplicate_parameters_for_clients(client_ids, phase='post_agg', reference_actor='server')
        # for cid in client_ids:
        #     self.result_manager.log_parameters(client_params, 'post_agg', cid)

        logger.info(f'[{self.name}] [Round: {self._round:03}] successfully aggregated into a new global model!')