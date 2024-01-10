from collections import defaultdict
import typing as t

import random

import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
import src.common.typing as fed_t
from src.strategy.abcstrategy import ABCStrategy
from src.config import *
from src.client.baseclient import BaseClient
from src.strategy.abcstrategy import *
from src.strategy.abcstrategy import StrategyIns

# Type declarations
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]

def passthrough_communication(ins: t.Any) -> t.Any:
    '''Simple passthrough communication logic'''
    return ins

def weighted_parameter_averaging(param_keys: t.Iterable, in_params: fed_t.ClientParams_t, weights: dict[str, float]) -> fed_t.ActorParams_t:

    out_params= {}
    for key in param_keys:
        temp_parameter = torch.Tensor()

        for cid, client_param in in_params.items():
            if temp_parameter.numel() == 0:
                temp_parameter =weights[cid] * client_param[key].data
            else:
                temp_parameter.data.add_(weights[cid] * client_param[key].data)
        
        out_params[key] = temp_parameter
    return out_params

def random_client_selection(sampling_fraction: float, cids: list[str]) -> fed_t.ClientIds_t:

    num_clients = len(cids)
    num_sampled_clients = max(int(sampling_fraction * num_clients), 1)
    sampled_client_ids = sorted(random.sample(cids, num_sampled_clients))

    return sampled_client_ids

def select_all_clients(cids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
    return cids


@dataclass
class StrategyCfgProtocol(t.Protocol):
    '''Protocol for base strategy config'''
    train_fraction: float
    eval_fraction: float

@dataclass
class BaseInsProtocol(t.Protocol):
    client_params: fed_t.ActorParams_t
    data_size: int

@dataclass
class BaseIns(StrategyIns):
    client_params: fed_t.ActorParams_t
    data_size: int

AllIns_t = dict[str, BaseIns]

@dataclass
class BaseOuts(StrategyOuts):
    server_params: fed_t.ActorParams_t


class BaseStrategy(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: StrategyCfgProtocol) -> None:
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        self._param_keys = self._server_params.keys()

        # self._client_params: ClientParams_t = defaultdict(dict)
        self._client_weights: dict[str, float] = defaultdict()
        self._client_ins: dict[str, float] = defaultdict()

    
    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> BaseOuts:
        base_outs = BaseOuts(
            server_params=ins.params,
        )
        return base_outs
    
    @classmethod
    def client_send_strategy(cls, ins: BaseInsProtocol, res: fed_t.Result) -> fed_t.ClientResult1:
        result =res
        result.size = ins.data_size
        return fed_t.ClientResult1(ins.client_params, res) 

    def receive_strategy(self, results: fed_t.ClientResults_t) -> AllIns_t:
        client_params = {}
        client_data_sizes = {}
        strat_ins = {}
        for cid, res in results.items():
            strat_ins[cid] = BaseIns(client_params=res.params, data_size=res.result.size)
        # strat_ins = BaseIns(client_params=client_params,
        #                    data_sizes=client_data_sizes)
        return strat_ins
    
    def send_strategy(self, ids: fed_t.ClientIds_t) -> fed_t.ClientIns_t:
        '''Simple send the same model to all clients strategy'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fed_t.ClientIns(
                params=self._server_params,
                metadata={}
            )
        return clients_ins
    
    def train_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)


    def aggregate(self, strategy_ins: AllIns_t) -> BaseOuts:
        # calculate client weights according to sample sizes
        total_size = sum([strat_in.data_size for strat_in in strategy_ins.values()])

        for cid, strat_in in strategy_ins.items():
            self._client_weights[cid] = float(strat_in.data_size / total_size)

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        # print((list(_client_params.values())[0].keys()))
        
        self._server_params = weighted_parameter_averaging(self._param_keys, _client_params, self._client_weights)

        outs = BaseOuts(server_params=self._server_params)

        return outs
    

FedAvgStrategy = BaseStrategy
