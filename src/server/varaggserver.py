import logging
import torch
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import parameters_to_vector
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from src.config import ClientConfig, VaraggServerConfig
# from .fedavgserver import FedavgServer
from src.results.resultmanager import ClientResult
from .baseserver import BaseServer, BaseOptimizer
from src.client.varaggclient import VaraggClient
from typing import Iterator, Tuple, Iterable
from torch.nn import Parameter
from torch import Tensor
import numpy as np
logger = logging.getLogger(__name__)


class VaraggOptimizer(BaseOptimizer):
    def __init__(self, params: Iterator[Parameter], param_keys: Iterable,  client_ids, **kwargs):
        self.lr = kwargs.get('lr')
        defaults = dict(lr=self.lr)
        super().__init__(params=params, defaults=defaults)

        # Gamma is the scaling coeffic
        self.gamma = kwargs.get('gamma')
        self.alpha = kwargs.get('alpha')
        self.local_grad_norm = None
        self.server_grad_norm = None

        self._importance_coefficients: dict[str, dict[str, Tensor]] = dict.fromkeys(client_ids, dict.fromkeys(param_keys, 1.0/len(client_ids)))
        self._server_params: list[Parameter] = self.param_groups[0]['params']
        self.param_keys = param_keys
        
    def _compute_scaled_weights(self, std_dict: dict[str, Parameter]) -> dict[str, Tensor]:
        weights = {}
        for key, val in std_dict.items():
            # ic(val)
            tanh_std = 1 - torch.tanh(val.data)
            # ic(tanh_std)
            weights[key] = tanh_std.mean()
            # ic(key, weights[key])
        return weights

    def _update_coefficients(self, client_id, importance: dict[str, Tensor]):
        client_coefficient = self._importance_coefficients[client_id]

        for name, weight_per_param in importance.items():
            client_coefficient[name]= self.alpha * client_coefficient[name] + (1 - self.alpha)* weight_per_param

        self._importance_coefficients[client_id] = client_coefficient


    def step(self, closure=None):
        # single step in a round of cgsv
        loss = None
        if closure is not None:
            loss = closure()
        # TODO: what to do if param groups are multiple
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # gradient
                delta = param.grad.data
                # FIXME: switch to an additive gradient with LR?
                # w = w - ∆w 
                param.data.sub_(delta)
        return loss

    def normalize_coefficients(self):
        total_coeff = dict.fromkeys(self.param_keys, 0.0)

        for client, coeff in self._importance_coefficients.items():
            for layer, tensor in coeff.items():
                total_coeff[layer] += tensor
        for client, coeff in self._importance_coefficients.items():
            for layer, tensor in coeff.items():
                assert total_coeff[layer] > 1e-9, f'Coefficient total is too small'
                self._importance_coefficients[client][layer] = tensor/total_coeff[layer]

        
        # assert 1.0 - sum(self._importance_coefficients.values()) < 1e-5


    def accumulate(self, client_params: dict[str, Parameter],client_id):
        # THis function is called per client. i.e. n clients means n calls
        # TODO: Rewrite this function to match gradient aggregate step
        # NOTE: Note that accumulate is called before step
        # NOTE: Currently supporting only one param group
        self._server_params: list[Parameter] = self.param_groups[0]['params']
        # ic(type(self._server_params))
        # # local_params = [param.data.float() for _, param in client_params.items()]
        # ic(self._server_params[0].get_device())
        # ic(list(client_params.values())[0].get_device())
        # ic(self._importance_coefficients[])


        local_grads = []
        server_grads = []
        i = 0
        for server_param, (name, local_param) in zip(self._server_params, client_params.items()):
                i += 1
                local_delta = server_param - local_param

                norm = local_delta.norm() 
                if norm == 0:
                    logger.warning(f"CLIENT [{client_id}]: Got a zero norm!")
                    local_grad_norm = local_delta.mul(self.gamma)
                else:
                    local_grad_norm = local_delta.div(norm).mul(self.gamma)

                # ic(local_grad_norm.get_device())
                # ic(self._importance_coefficients[client_id][name].get_device())
                weighted_local_grad = local_grad_norm.mul(self._importance_coefficients[client_id][name])
                
                # server params grad is used as a buffer
                if server_param.grad is None:
                    server_param.grad = weighted_local_grad
                else:
                    server_param.grad.add_(weighted_local_grad)

                server_grads.append(server_param.grad.data)
                local_grads.append(local_grad_norm.data)

@dataclass
class PerClientResult:
    param: dict[str, np.ndarray] =  field(default_factory=dict)
    param_std: dict[str, np.ndarray] =  field(default_factory=dict)
    std_weights: dict[str, np.ndarray] =  field(default_factory=dict)


@dataclass
class VaraggResult:
    round: int
    clients: dict[str, PerClientResult]
    server_params: dict[str, np.ndarray]
    imp_coeffs : dict[str, dict[str, np.ndarray]]

class VaraggServer(BaseServer):
    name:str = 'VaraggServer'

    def __init__(self, cfg:VaraggServerConfig, *args, **kwargs):
        self.clients: dict[str, VaraggClient]

        super(VaraggServer, self).__init__(cfg, *args, **kwargs)
        self.round = 0
        self.cfg = cfg
        
        self.importance_coefficients = dict.fromkeys(self.clients, 0.0)

        self.server_optimizer: VaraggOptimizer = VaraggOptimizer(params=self.model.parameters(),
         param_keys=dict(self.model.named_parameters()).keys(), client_ids=self.clients.keys(),
         lr= self.client_cfg.lr, gamma=self.cfg.gamma, alpha=self.cfg.alpha)

        # lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)

    @staticmethod
    def detorch_params(param_dict: dict[str, Parameter]):
        out_dict = {}
        for name, param in param_dict.items():
            out_dict[name] = param.detach().cpu().abs().mean().item()
        return out_dict

    def _aggregate(self, ids, train_results: ClientResult):
        
        # Calls client upload and server accumulate
        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer

        client_results = {idx: PerClientResult() for idx in ids}

        # Compute coefficients
        for identifier in ids:
            client_params_std = self.clients[identifier].parameter_std_dev()
            new_weights = self.server_optimizer._compute_scaled_weights(client_params_std)
            self.server_optimizer._update_coefficients(identifier, new_weights)
            client_results[identifier].param_std = self.detorch_params(client_params_std)
            client_results[identifier].std_weights = self.detorch_params(new_weights)

        self.server_optimizer.normalize_coefficients()

        # accumulate weights

        for identifier in ids: 
            client_params = self.clients[identifier].upload()
            self.server_optimizer.accumulate(client_params, identifier)
            client_results[identifier].param = self.detorch_params(client_params)


        self.server_optimizer.step()
        self.lr_scheduler.step() # update learning rate

        server_params_np = self.detorch_params(dict(self.model.named_parameters()))
        imp_coeffs = {}
        for cl, coeffs in self.server_optimizer._importance_coefficients.items():
            imp_coeffs[cl] = self.detorch_params(coeffs)
        varag_result =  VaraggResult(clients=client_results, server_params=server_params_np, imp_coeffs=imp_coeffs, round=self.round)

        self.result_manager.save_as_csv(asdict(varag_result), filename='varag_results.csv')

        logger.info(f'[{self.name}] [Round: {self.round:03}] ...successfully aggregated into a new global model!')
