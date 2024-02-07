import logging
from collections import OrderedDict
import typing as t
from dataclasses import dataclass
from functools import partial
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CosineSimilarity, Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.functional import cosine_similarity, tanh
from feduciary.config import ClientConfig, CGSVConfig
from feduciary.results.resultmanager import ResultManager
from feduciary.strategy import *
import feduciary.common.typing as fed_t
from feduciary.common.utils import generate_client_ids
from feduciary.strategy.fedoptstrategy import compute_server_delta_w_normalize, compute_server_delta, add_server_deltas
# from feduciary.strategy.fedstdevstrategy import normalize_coefficients

logger = logging.getLogger(__name__)


@dataclass
class CgsvCfgProtocol(t.Protocol):
    train_fraction: float
    eval_fraction: float
    num_clients: int
    lr: float
    gamma: float
    alpha: float 
    delta_normalize: bool

@dataclass
class CgsvIns(StrategyIns):
    pass

AllIns_t = dict[str, CgsvIns]

@dataclass
class CgsvOuts(StrategyOuts):
    client_params: fed_t.ClientParams_t

def compute_cgsv(server_delta: fed_t.ActorDeltas_t, local_delta: fed_t.ActorDeltas_t) -> float:
    server_delta_vec = parameters_to_vector(list(server_delta.values()))
    local_delta_vec = parameters_to_vector(list(local_delta.values()))

    return cosine_similarity(server_delta_vec, local_delta_vec, dim=0).item()


def add_momentum(weight: float, cgsv: float, alpha: float) -> float:
    return alpha * weight + (1 - alpha)* cgsv


class CgsvStrategy(ABCStrategy):
    name: str = 'CgsvStrategy'

    def __init__(self, model: Module,
                cfg: CgsvCfgProtocol,
                res_man: ResultManager):
        
        self.cfg = cfg
        self.res_man = res_man
        client_ids = generate_client_ids(cfg.num_clients)

        if self.cfg.delta_normalize:
            self._update_fn = partial(compute_server_delta_w_normalize, gamma=self.cfg.gamma)
        else:
            self._update_fn = compute_server_delta

        self._server_params: dict[str, Parameter] = model.state_dict()

        
        self.local_grad_norm = None
        self.server_grad_norm = None

        # self._cos_sim = CosineSimilarity(dim=0)
        # NOTE: Differing in initialization here from paper as it leads to permanently zero gradients
        self._client_wts = {cid: 1.0/len(client_ids) for cid in client_ids}
        self._omegas = {cid: 1.0/len(client_ids) for cid in client_ids}

        self._client_params = {cid: model.state_dict() for cid in client_ids}


    def receive_strategy(self, ins: fed_t.ClientResults_t) -> AllIns_t:
        return {cid: CgsvIns(cl_res.params) for cid, cl_res in ins.items()}

    def send_strategy(self, ids: fed_t.ClientIds_t) -> fed_t.ClientIns_t:
        '''Send custom models to respective clients'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fed_t.ClientIns(
                params=self._client_params[cid],
                metadata={}
            )
        return clients_ins


    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> ClientInProto:
        return  ClientInProto(in_params=ins.params)

    @classmethod
    def client_send_strategy(cls, ins: CgsvIns, result: fed_t.Result) -> fed_t.ClientResult:
        return fed_t.ClientResult(ins.client_params, result) 

    def train_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)
    


    
    def _mask(self, params, q):
        return {k: v for k, v in params.items() if k in q}
    
    def _sparsify_gradients(self, client_ids, wts):
        q = {}
        v = {}
        for cid in client_ids:
            q[cid] = self._client_wts[cid] * self._omegas[cid] 
            v[cid] = self._client_wts[cid] * self._omegas[cid]
        return v, q

    @classmethod
    def normalize_weights(cls, in_weights: dict[str, float]):
        total = 1e-9
        out_weights ={}
        for val in in_weights.values():
            total += val

        for key, val in in_weights.items():
            out_weights[key] = val/total
        
        return out_weights
    
    def aggregate(self, strategy_ins: AllClientIns_t) -> CgsvOuts:

        client_ids = list(strategy_ins.keys())
        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}

        server_delta, client_deltas = self._update_fn(self._server_params, _client_params, self._client_wts)
        # k1 = list(server_delta.keys())[0]
        # ic(server_delta[k1].device)

        for cid in client_ids:
            # ic(client_deltas[cid][k1].device)

            cgsv =  compute_cgsv(server_delta, client_deltas[cid])
            self._client_wts[cid] = add_momentum(self._client_wts[cid], cgsv, self.cfg.alpha)


        # Normalize the coefficients
        self._client_wts = self.normalize_weights(self._client_wts)

        # Logging code
        self.res_man.log_general_metric(self._client_wts, phase='post_agg', actor='server', metric_name='client_weights')

        self.res_man.log_parameters(self._server_params, phase='post_agg', actor='server')
        
        
        for cid in client_ids:
            pass
        # print(self._client_wts)
        # print(self._client_wts.values())
        # print(self._client_wts.keys
        return CgsvOuts(server_params=self._server_params,
                        client_params=self._client_params)

