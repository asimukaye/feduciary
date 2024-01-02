from collections import OrderedDict, defaultdict
import typing as t

import random

import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
from src.common.typing import ClientIns, ClientResult1, Result 
from src.strategy.abcstrategy import ABCStrategy
from src.config import *
from src.client.baseclient import BaseClient
from src.strategy.abcstrategy import *
from src.common.typing import ClientIns, ClientIns_t, ClientResults_t
from src.strategy.abcstrategy import StrategyIns
# Type declarations
# ClientParams_t = dict[str, OrderedDict[str, Parameter]]
# ActorParams_t = OrderedDict[str, Parameter]
# Clients_t = dict[str, BaseClient]
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]

def passthrough_communication(ins: t.Any) -> t.Any:
    '''Simple passthrough communication logic'''
    return ins

def weighted_parameter_averaging(param_keys: t.Iterable, in_params: ClientParams_t, weights: dict[str, float]) -> ActorParams_t:

    out_params: ActorParams_t = OrderedDict()
    for key in param_keys:
        temp_parameter = torch.Tensor()

        for cid, client_param in in_params.items():
            if temp_parameter.numel() == 0:
                temp_parameter =weights[cid] * client_param[key].data
            else:
                temp_parameter.data.add_(weights[cid] * client_param[key].data)
        
        out_params[key].data = temp_parameter.data
    return out_params

def random_client_selection(sampling_fraction: float, cids: list[str]) -> ClientIds_t:

    num_clients = len(cids)
    num_sampled_clients = max(int(sampling_fraction * num_clients), 1)
    sampled_client_ids = sorted(random.sample(cids, num_sampled_clients))

    return sampled_client_ids

def select_all_clients(cids: ClientIds_t) -> ClientIds_t:
    return cids


@dataclass
class StrategyCfgProtocol(t.Protocol):
    '''Protocol for base strategy config'''
    train_fraction: float
    eval_fraction: float

# @dataclass
# class BaseIns(StrategyIns):
#     client_params: ClientParams_t
#     data_sizes: dict[str, int]
@dataclass
class BaseIns(StrategyIns):
    client_params: ActorParams_t
    data_size: int

AllIns_t = dict[str, BaseIns]

@dataclass
class BaseOuts(StrategyOuts):
    server_params: ActorParams_t


class BaseStrategy(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: StrategyCfgProtocol) -> None:
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: OrderedDict[str, Parameter] = OrderedDict(model.state_dict())
        self._param_keys = self._server_params.keys()

        # self._client_params: ClientParams_t = defaultdict(dict)
        self._client_weights: dict[str, float] = defaultdict()
        self._client_ins: dict[str, float] = defaultdict()

    
    @classmethod
    def client_receive_strategy(cls, ins: ClientIns) -> BaseOuts:
        base_outs = BaseOuts(
            server_params=ins.params,
        )
        return base_outs
    
    @classmethod
    def client_send_strategy(cls, ins: BaseIns) -> ClientResult1:
        # TODO: Decide on the protocol for this later
        return ClientResult1(ins.client_params, Result(size=ins.data_size)) 

    def receive_strategy(self, results: ClientResults_t) -> AllIns_t:
        client_params = {}
        client_data_sizes = {}
        strat_ins = {}
        for cid, res in results.items():
            strat_ins[cid] = BaseIns(client_params=res.params, data_size=res.result.size)
        # strat_ins = BaseIns(client_params=client_params,
        #                    data_sizes=client_data_sizes)
        return strat_ins
    
    def send_strategy(self, ids: ClientIds_t) -> ClientIns_t:
        '''Simple send the same model to all clients strategy'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = ClientIns(
                params=self._server_params,
                metadata=OrderedDict()
            )
        return clients_ins
    
    def train_selection(self, in_ids: ClientIds_t) -> ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: ClientIds_t) -> ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)


    def aggregate(self, strategy_ins: AllIns_t) -> BaseOuts:
        # calculate client weights according to sample sizes
        total_size = sum([strat_in.data_size for strat_in in strategy_ins.values()])

        for cid, strat_in in strategy_ins.items():
            self._client_weights[cid] = float(strat_in.data_size / total_size)

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        
        self._server_params = weighted_parameter_averaging(self._param_keys, _client_params, self._client_weights)

        outs = BaseOuts(server_params=self._server_params)

        return outs
    

FedAvgStrategy = BaseStrategy
