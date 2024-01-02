from collections import OrderedDict, defaultdict
import typing as t
from typing import Any

import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim 
from src.strategy.abcstrategy import ABCStrategy
from src.strategy.basestrategy import passthrough_communication, random_client_selection
import src.common.typing as fed_t

from src.config import *
from src.client.baseclient import BaseClient

import random
# Type declarations
ClientParams_t = dict[str, OrderedDict[str, Parameter]]
ActorParams_t = OrderedDict[str, Parameter]
ServerDeltas_t = OrderedDict[str, Tensor]

Clients_t = dict[str, BaseClient]
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]

# FIXME: !!!!!!

def gradient_average_update(server_params: ActorParams_t,
                            client_params: ClientParams_t,
                            weights: ScalarWeights_t) -> tuple[ActorParams_t, ServerDeltas_t]:
    
    server_deltas: ServerDeltas_t = {}
    
    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            client_delta = client_param[key].data - server_param.data

            if server_deltas[key] is None:
                server_deltas[key] = weights[cid] * client_delta
            else:
                server_deltas[key].add_(weights[cid] * client_delta)

    for key, delta in server_deltas.items():
        server_params[key].data.add_(delta)
    
    return server_params, server_deltas


def gradient_average_with_delta_normalize(server_params: ActorParams_t,
                            client_params: ClientParams_t,
                            weights: ScalarWeights_t,
                            gamma: float) -> tuple[ActorParams_t, ServerDeltas_t]:
    
    def _delta_normalize(delta: Tensor, gamma: float) -> Tensor:
        '''Normalize the parameter delta update and scale to prevent potential gradient explosion'''
        norm = delta.norm() 
        if norm == 0:
            logger.warning(f"Normalize update: Got a zero norm update")
            delta_norm = delta.mul(gamma)
        else:
            delta_norm = delta.div(norm).mul(gamma)
        return delta_norm
    

    server_deltas: ServerDeltas_t = {}

    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            # client delta =  client param(w_k+1,i) - server param (w_k)
            client_delta = client_param[key].data - server_param.data
            
            client_delta = _delta_normalize(client_delta, gamma)

            if server_deltas[key] is None:
                server_deltas[key] = weights[cid] * client_delta
            else:
                server_deltas[key].add_(weights[cid] * client_delta)

    for key, delta in server_deltas.items():
        server_params[key].data.add_(delta)

    return server_params, server_deltas

@dataclass
class FedOptCfgProtocol(t.Protocol):
    '''Protocol for base strategy config'''
    train_fraction: float
    eval_fraction: float
    delta_normalize: bool
    gamma: float


class FedOptStrategy(ABCStrategy):
    def __init__(self, model: Module,cfg: FedOptCfgProtocol, client_lr: float) -> None:
        # super().__init__(model, client_lr, cfg)
        self.cfg = cfg
        defaults = dict(lr=client_lr)

        self.server_optimizer = torch.optim.Optimizer(model.parameters(), defaults)
        assert len(self.server_optimizer.param_groups) == 1, f'Multi param group yet to be implemented'
        self._server_params: OrderedDict[str, Parameter] = model.state_dict()
        self._server_deltas: OrderedDict[str, Tensor] = {param:torch.tensor(0.0) for param in self._server_params.keys()}

        self._client_params: ClientParams_t = defaultdict(dict)
        self._client_weights: dict[str, float] = defaultdict()

    def send_strategy(self, ins: Any) -> Any:
        return passthrough_communication(ins)
    
    def receive_strategy(self, ins: Any) -> Any:
        return passthrough_communication(ins)
    
    def train_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)


    def aggregate(self, client_data_sizes: dict[str, int]):
        # calculate client weights according to sample sizes
        self._client_weights = {}
        for cid, data_size in client_data_sizes.items():
            self._client_weights[cid] = float(data_size / sum(client_data_sizes.values())) 
