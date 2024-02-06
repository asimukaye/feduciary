from collections import defaultdict
import typing as t
import random
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
import feduciary.common.typing as fed_t
# from feduciary.strategy.abcstrategy import ABCStrategy
from feduciary.strategy.abcstrategy import *
from feduciary.strategy.basestrategy import passthrough_communication, random_client_selection, weighted_parameter_averaging
from feduciary.common.utils import generate_client_ids

from feduciary.results.resultmanager import ResultManager

# Type declarations
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]

@dataclass
class StrategyCfgProtocol(t.Protocol):
    '''Protocol for base strategy config'''
    train_fraction: float
    eval_fraction: float
    weights: list[float]
    num_clients: int


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


class FedavgManual(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: StrategyCfgProtocol,
                 res_man: ResultManager) -> None:
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        self._param_keys = self._server_params.keys()

        # self._client_params: ClientParams_t = defaultdict(dict)
        client_ids = generate_client_ids(cfg.num_clients)
        self._client_weights: dict[str, float] = {cid: wt for cid, wt in zip(client_ids, cfg.weights)}

        self._client_ins: dict[str, float] = defaultdict()


    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> BaseOuts:
        base_outs = BaseOuts(
            server_params=ins.params,
        )
        return base_outs
    
    @classmethod
    def client_send_strategy(cls, ins: BaseInsProtocol, res: fed_t.Result) -> fed_t.ClientResult:
        result =res
        result.size = ins.data_size
        return fed_t.ClientResult(ins.client_params, res) 

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
        # calculate client weights according to config
        # ic(self._client_weights)
 
        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        
        self._server_params = weighted_parameter_averaging(self._param_keys, _client_params, self._client_weights)

        outs = BaseOuts(server_params=self._server_params)

        return outs
    
