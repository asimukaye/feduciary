from collections import defaultdict
import typing as t
from typing import Any
# from dataclasses import dataclass, field
from functools import partial
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim 
from feduciary.strategy.abcstrategy import *
from feduciary.strategy.basestrategy import passthrough_communication, random_client_selection, ClientInProto
import feduciary.common.typing as fed_t

# Type declarations

ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]

def add_param_deltas(server_params: fed_t.ActorParams_t,
                      server_deltas: fed_t.ActorDeltas_t) -> fed_t.ActorParams_t:
    '''Add deltas to the server parameters'''
    for key, delta in server_deltas.items():
        server_params[key].data.add_(delta)
    return server_params


def compute_server_delta(server_params: fed_t.ActorParams_t,
                            client_params: fed_t.ClientParams_t,
                            weights: ScalarWeights_t) -> tuple[fed_t.ActorDeltas_t, fed_t.ClientDeltas_t]:
    server_deltas: fed_t.ActorDeltas_t = {}
    client_deltas: fed_t.ClientDeltas_t = defaultdict(dict)
    
    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            client_delta = client_param[key].data - server_param.data
            server_deltas[key] = server_deltas.get(key, 0) + weights[cid] * client_delta
            client_deltas[cid][key] = client_delta

    return server_deltas, client_deltas

def gradient_average_update(server_params: fed_t.ActorParams_t,
                            client_params: fed_t.ClientParams_t,
                            weights: ScalarWeights_t) -> tuple[fed_t.ActorParams_t, fed_t.ActorDeltas_t]:
    
    server_deltas, _ = compute_server_delta(server_params, client_params, weights)
    server_params = add_param_deltas(server_params, server_deltas)
    
    return server_params, server_deltas

def _delta_normalize(delta: Tensor, gamma: float) -> Tensor:
    '''Normalize the parameter delta update and scale to prevent potential gradient explosion'''
    norm = delta.norm() 
    if norm == 0:
        logger.warning(f"Normalize update: Got a zero norm update")
        delta_norm = delta.mul(gamma)
    else:
        delta_norm = delta.div(norm).mul(gamma)
    return delta_norm
  

def compute_server_delta_w_normalize(server_params: fed_t.ActorParams_t,
                            client_params: fed_t.ClientParams_t,
                            weights: ScalarWeights_t,
                            gamma: float) -> tuple[fed_t.ActorDeltas_t, fed_t.ClientDeltas_t]:
    server_deltas: fed_t.ActorDeltas_t = {}
    client_deltas: fed_t.ClientDeltas_t = defaultdict(dict)
    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            # client delta =  client param(w_k+1,i) - server param (w_k)
            client_delta = client_param[key].data - server_param.data
            client_delta = _delta_normalize(client_delta, gamma)
            server_deltas[key] = server_deltas.get(key, 0) + weights[cid] * client_delta
            client_deltas[cid][key] = client_delta

    return server_deltas, client_deltas


def gradient_average_with_delta_normalize(server_params: fed_t.ActorParams_t,
                            client_params: fed_t.ClientParams_t,
                            weights: ScalarWeights_t,
                            gamma: float) -> tuple[fed_t.ActorParams_t, fed_t.ActorDeltas_t]:
    server_deltas, _ = compute_server_delta_w_normalize(server_params, client_params, weights, gamma)
    server_params = add_param_deltas(server_params, server_deltas)
    return server_params, server_deltas

@dataclass
class FedOptCfgProtocol(t.Protocol):
    '''Protocol for base strategy config'''
    train_fraction: float
    eval_fraction: float
    delta_normalize: bool
    gamma: float


@dataclass
class FedOptIns(StrategyIns):
    data_size: int

AllIns_t = dict[str, FedOptIns]
@dataclass
class FedOptOuts(StrategyOuts):
    pass


class FedOptStrategy(ABCStrategy):
    def __init__(self, model: Module,
                 cfg: FedOptCfgProtocol,
                 res_man: ResultManager,
                 ) -> None:
        # super().__init__(model, client_lr, cfg)
        self.cfg = cfg
        self.res_man = res_man
        # TBD: Server Optimizer style to be implemented
        # defaults = dict(lr=client_lr)

        # self.server_optimizer = torch.optim.Optimizer(model.parameters(), defaults)
        # assert len(self.server_optimizer.param_groups) == 1, f'Multi param group yet to be implemented'


        self._server_params: dict[str, Parameter] = model.state_dict()
        self._server_deltas: dict[str, Tensor] = {param:torch.tensor(0.0) for param in self._server_params.keys()}

        if self.cfg.delta_normalize:
            self._update_fn = partial(gradient_average_with_delta_normalize, gamma=self.cfg.gamma)
        else:
            self._update_fn = gradient_average_update

        self._client_params: fed_t.ClientParams_t = defaultdict(dict)
        self._client_wts: dict[str, float] = defaultdict()

    def send_strategy(self, ids: fed_t.ClientIds_t) -> fed_t.ClientIns_t:
        return {cid: fed_t.ClientIns(params=self._server_params,
                                    metadata={}) for cid in ids}
    
    def receive_strategy(self, ins: fed_t.ClientResults_t) -> AllClientIns_t:
        return {cid: FedOptIns(cl_res.params, cl_res.result.size) for cid, cl_res in ins.items()}
    
    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> ClientInProto:
        return  ClientInProto(in_params=ins.params)
    
    @classmethod
    def client_send_strategy(cls, ins: FedOptIns, result: fed_t.Result) -> fed_t.ClientResult:
        return fed_t.ClientResult(ins.client_params, result)

    
    def train_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)


    def aggregate(self, strategy_ins: AllIns_t) -> FedOptOuts:
        # calculate client weights according to sample sizes
        total_size = sum([strat_in.data_size for strat_in in strategy_ins.values()])

        for cid, strat_in in strategy_ins.items():
            self._client_wts[cid] = float(strat_in.data_size / total_size)

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}

        self._server_params,self._server_deltas = self._update_fn(self._server_params, _client_params, self._client_wts)

                # for cid in client_ids:
        self.res_man.log_general_metric(self._client_wts, phase='post_agg', actor='server', metric_name='client_weights')
        self.res_man.log_parameters(self._server_params, phase='post_agg',
                                    actor='server')

        return FedOptOuts(server_params=self._server_params)
        
