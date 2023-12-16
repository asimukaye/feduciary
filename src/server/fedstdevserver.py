import logging
import torch
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import parameters_to_vector
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from src.config import FedstdevServerConfig
from src.utils import ClientParams_t
from collections import defaultdict
from src.results.resultmanager import ResultManager
from .baseserver import BaseServer, BaseStrategy
from src.client.fedstdevclient import FedstdevClient
import typing as t
from torch.nn import Parameter, Module
from torch import Tensor
import numpy as np  
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class PerClientResult:
    param: dict[str, np.ndarray] =  field(default_factory=dict)
    param_std: dict[str, np.ndarray] =  field(default_factory=dict)
    std_weights: dict[str, np.ndarray] =  field(default_factory=dict)

@dataclass
class FedstdevResult:
    round: int
    clients: dict[str, PerClientResult]
    server_params: dict[str, np.ndarray]
    imp_coeffs : dict[str, dict[str, np.ndarray]]

ClientParamWeight_t= t.Dict[str, dict[str, float]] 

class FedstdevOptimizer(BaseStrategy):
    def __init__(self, model: Module, client_lr: float, cfg: FedstdevServerConfig, client_ids: list[str], res_man: ResultManager = None):
        super().__init__(model, client_lr, cfg)
        self.cfg = cfg

        # Giving a reference to result manager for internal logging of states
        self.res_man = res_man

        self.gamma = cfg.gamma
        self.alpha = cfg.alpha
        param_keys = self._server_params.keys()
        self.param_keys = param_keys

        betas = cfg.betas
        assert len(param_keys)==len(betas)
        self.local_grad_norm = None
        self.server_grad_norm = None
        self.beta = {param: beta for param, beta in zip(param_keys, betas)}
  
        # TODO: Think of ways to support dynamic client allocation
        per_client_per_param_imp =  1.0/len(client_ids)
        self._client_weights: ClientParamWeight_t = {cid: {param: per_client_per_param_imp for param in param_keys} for cid in client_ids}

        # NOTE: IMPORTANT: Dict.fromkeys() creates copies with same memory, bad idea for creating nested dictionaries
        # self._client_weights: ClientParamWeight_t  = dict.fromkeys(client_ids, dict.fromkeys(param_keys, per_client_per_param_imp))

        self._client_omegas: ClientParamWeight_t = {cid: {param: per_client_per_param_imp for param in param_keys} for cid in client_ids}


        # Additional dictionaries required for this approach
        self._client_params_std: ClientParams_t = {cid: {param: None for param in param_keys} for cid in client_ids}
        # tracking client deltas for logging
        self._client_deltas: ClientParams_t = {cid: {param: None for param in param_keys} for cid in client_ids}

        
    def _compute_scaled_weights(self, std_dict: dict[str, Parameter]) -> dict[str, float]:
        weights = {}
        for key, val in std_dict.items():
            # ic(val)
            sigma_scaled = self.beta[key]*val.data
            tanh_std = 1 - torch.tanh(sigma_scaled)
            # ic(tanh_std)
            weights[key] = tanh_std.mean().item()
            # ic(key, weights[key])
        return weights

    def _compute_mean_weights(self, std_dict: dict[str, Parameter]) -> dict[str, float]:
        sigma_scaled = {}
        for key, val in std_dict.items():
            sigma_scaled[key] = self.beta[key]*val.data.mean().item()
        return sigma_scaled            
       
    def _min_max_normalized_weights(self, in_weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out_weights = defaultdict(dict)
        temp_weight = defaultdict(list)
        for id, weights in in_weights.items():
            for layer, weight in weights.items():
                temp_weight[layer].append(weight)
        
        for id, weights in in_weights.items():
            for layer, weight in weights.items():
                out_weights[id][layer] = 1 - (weight - min(temp_weight[layer]))/(max(temp_weight[layer])- min(temp_weight[layer]))
        
        return out_weights

    def _delta_normalize(self, delta: Tensor, gamma: float) -> Tensor:
        '''Normalize the parameter delta update and scale to prevent potential gradient explosion'''
        norm = delta.norm() 
        if norm == 0:
            logger.warning(f"Normalize update: Got a zero norm update")
            delta_norm = delta.mul(gamma)
        else:
            delta_norm = delta.div(norm).mul(gamma)
        return delta_norm
    
    def _add_weight_momentum(self, client_id, omegas: dict[str, Tensor]):
        
        client_wt = self._client_weights[client_id]
        # Avoiding reading and writing to a dictionary simultaneously

        for name, omega_per_param in omegas.items():
            client_wt[name] = self.alpha * client_wt[name] + (1 - self.alpha)* omega_per_param

        self._client_weights[client_id] = client_wt


    def _normalize_weights(self):
        total_coeff = {param:0.0 for param in self.param_keys}

        for cid, coeff in self._client_weights.items():
            for layer, tensor in coeff.items():
                total_coeff[layer] += tensor
        for cid, coeff in self._client_weights.items():
            for layer, tensor in coeff.items():
                assert total_coeff[layer] > 1e-9, f'Coefficient total is too small'
                self._client_weights[cid][layer] = tensor/total_coeff[layer]



        
    def set_client_param_stds(self, cid, param_std: OrderedDict)-> None:
        self._client_params_std[cid] = param_std


    def param_update_rule(self) -> None:
        # Standard gradient accumulation rule

        for key, server_param in self._server_params.items():
            for cid, client_param in self._client_params.items():
                # Using FedNova Notation of delta (Δ) as (-grad ∇)
                # client delta =  client param(w_k+1,i) - server param (w_k)
                client_delta = client_param[key].data.sub(server_param.data)              
                if self.cfg.delta_normalize:
                    client_delta = self._delta_normalize(client_delta, self.gamma)

                self._client_deltas[cid][key] = client_delta
                if self._server_deltas[key] is None:
                    self._server_deltas[key] = self._client_weights[cid][key] * client_delta
                else:
                    self._server_deltas[key].add_(self._client_weights[cid][key] * client_delta)

        for key, delta in self._server_deltas.items():
            self._server_params[key].data.add_(delta)
        
        # LOGGING CODE
        for cid, cl_delta in self._client_deltas.items():
            self.res_man.log_parameters(cl_delta, 'post_agg', cid, metric='param_delta', verbose=True)

        self.res_man.log_parameters(self._server_deltas, 'post_agg', 'server', metric='param_delta', verbose=True)


    def aggregate(self, client_ids):
        
        if self.cfg.weight_scaling == 'tanh':
            for cid in client_ids:
                omega = self._compute_scaled_weights(self._client_params_std[cid])
                self._add_weight_momentum(cid, omega)
                self._client_omegas[cid] = omega

        elif self.cfg.weight_scaling == 'min_max':
            # Normalize twice version
            int_weights = {}
            for cid in client_ids:
                int_weights[cid] = self._compute_mean_weights(self._client_params_std[cid])

            normalized_weights = self._min_max_normalized_weights(int_weights)
            for cid in client_ids:
                self._add_weight_momentum(cid, normalized_weights[cid])
       
        else:
            logger.error(f'Unknown weight scaling type: {self.cfg.weight_scaling}')

        self._normalize_weights()

        self.res_man.log_general_metric(self._client_omegas, 'omegas', 'server', 'post_agg')


class FedstdevServer(BaseServer):
    name:str = 'FedstdevServer'

    def __init__(self, cfg: FedstdevServerConfig, *args, **kwargs):
        # Redeclaring clients just to make use of IDE prompting for fedstdev client
        self.clients: dict[str, FedstdevClient]

        super(FedstdevServer, self).__init__(cfg, *args, **kwargs)
        self.round = 0
        self.cfg = cfg
        

        self.server_optimizer: FedstdevOptimizer = FedstdevOptimizer(model=self.model, client_lr=self.client_cfg.lr, cfg=cfg, client_ids=list(self.clients.keys()), res_man=self.result_manager)
        
        # lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)


    @staticmethod
    def detorch_params(param_dict: dict[str, Parameter]):
        out_dict = {}
        for name, param in param_dict.items():
            out_dict[name] = param.detach().cpu().abs().mean().item()
        return out_dict

    @staticmethod
    def detorch_params_no_reduce(param_dict: dict[str, Parameter]) ->dict[str, list]:
        out_dict = {}
        for name, param in param_dict.items():
            out_dict[name] = param.detach().cpu().numpy()
        return out_dict
    
    def _compute_delta_sigma(self, del_dict: dict[str, np.ndarray], std_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # def _compute_delta_sigma(self, del_dict: dict[str, list], std_dict: dict[str, list]) -> dict[str, list]:
        out_dict = {}
        for name, delta in del_dict.items():
            out_dict[name] = np.abs(delta)*std_dict[name]
        return out_dict

    def save_full_param_dict(self, clients_mu:dict[str, dict], clients_delta: dict[str, dict], clients_std: dict[str, dict]):
        result = {}
        result['round'] = self.round
        result['clients_delta'] = clients_delta
        # ic(clients_delta.keys())
        result['clients_mu'] = clients_mu
        result['clients_std'] = clients_std
        clients_del_sigma = {}
        for clnt, delta in clients_delta.items():
            clients_del_sigma[clnt] = self._compute_delta_sigma(delta, clients_std[clnt])
        result['clients_del_sigma'] = clients_del_sigma

        if not os.path.exists('fedstdev_debug'):
            os.makedirs('fedstdev_debug')
        
        # with open(f'fedstdev_debug/fedstdev_full_{self.round}.json', 'w') as f:
        #     json.dump(result, f, indent=4)
        # ic(result['clients_delta'])
        np.savez_compressed(f'fedstdev_debug/fedstdev_{self.round:03}.npz', **result)
        # self.result_manager.save_as_csv(result, 'fedstdev_full.csv')


    def _run_strategy(self, client_ids: list[str], train_results):

        self.result_manager.log_parameters(self.model.state_dict(), phase='pre_agg', actor='server', verbose=True)

        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer

        for cid in client_ids:
            client_params = self.clients[cid].upload()
            self.server_optimizer.set_client_params(cid, client_params)
            client_param_stds = self.clients[cid].parameter_std_dev()
            self.server_optimizer.set_client_param_stds(cid, client_param_stds)

            self.result_manager.log_parameters(client_params, phase='pre_agg', actor=cid, verbose=True)
            self.result_manager.log_parameters(client_params, phase='pre_agg', actor=cid, verbose=True, metric='param_std')

    
        # if self.round in [0, 10, 20, 30, 50, 75, 100, 125, 149]:
        #     self.save_full_param_dict(clients_mu=client_mus, clients_delta=client_deltas, clients_std=client_stds)
            

        self.server_optimizer.aggregate(client_ids)
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # Full parameter debugging
        self.result_manager.log_general_metric(self.server_optimizer._client_weights, metric_name='client_weights', phase='post_agg', actor='server')

        self.result_manager.log_parameters(self.model.state_dict(), phase='post_agg', actor='server', verbose=True)
        self.result_manager.log_duplicate_parameters_for_clients(client_ids, phase='post_agg', reference_actor='server')
  
        logger.info(f'[{self.name}] [Round: {self.round:03}] successfully aggregated into a new global model!')
