import logging
from collections import OrderedDict
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CosineSimilarity, Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.nn.functional import cosine_similarity, tanh
from feduciary.config import ClientConfig, CGSVConfig
from feduciary.results.resultmanager import ResultManager
from feduciary.strategy import *
import feduciary.common.typing as fed_t
from feduciary.common.utils import generate_client_ids
from feduciary.strategy.fedoptstrategy import compute_server_delta_w_normalize, compute_server_delta, add_param_deltas
# from feduciary.strategy.fedstdevstrategy import normalize_coefficients
import numpy as np
logger = logging.getLogger(__name__)

# NOTE: THIS ALGORITHM HAS STABILITY ISSUES IN CASE THE LEARNING DOES NOT CONVERGE. If the cgsv values stay consistently negative, it would lead to the weights turning negative rendering a lot of subsequent steps invalid. This is a known issue with the algorithm and is not addressed in the paper.

@dataclass
class CgsvCfgProtocol(t.Protocol):
    train_fraction: float
    eval_fraction: float
    num_clients: int
    lr: float
    gamma: float
    alpha: float
    beta: float
    delta_normalize: bool
    sparsify_gradients: bool  

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


def add_momentum_and_clip(weight: float, cgsv: float, alpha: float) -> float:
    return max(1e-5, alpha * weight + (1 - alpha)* cgsv)


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
            logger.info('Using normalized delta')
        else:
            self._update_fn = compute_server_delta
            logger.info('Using non-normalized delta')

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
    

    # def zero_mask(self, delta, q):
    #     mask = torch.zeros_like(delta, dtype=torch.bool)
    #     mask[:int(q)] = 1
    #     return mask
    
    def _sparsify_gradients(self, wts: ScalarWeights_t,
                            server_deltas: fed_t.ActorDeltas_t,
                            beta: float) -> tuple[dict[str, fed_t.ActorDeltas_t], dict[str, int]]:
        """
        Sparsify the gradients based on the given weights, server deltas, and beta value.

        Args:
            wts (ScalarWeights_t): Dictionary of weights.
            server_deltas (fed_t.ActorDeltas_t): Dictionary of server deltas.
            beta (float): Beta value.

        Returns:
            Tuple: A tuple containing the sparsified deltas and the q values.
        """
        delta_vec = parameters_to_vector(list(server_deltas.values()))
        keys = list(server_deltas.keys())
        D = delta_vec.shape[0]

        cids = list(wts.keys())

        # ic(wts)
        tanh_br = {cid: np.tanh(wt * beta) for cid, wt in wts.items()}
        ic(tanh_br)
        # if tanh_br.values() == float('nan'):
        #     return None, None
        # tah_br = np.tanh(beta*list(wts.values())) 
        max_tanh_br = max(list(tanh_br.values()))
        q = {cid : int(D*tbr/max_tanh_br) for cid, tbr in tanh_br.items()}

        zeroed_deltas = {cid: delta_vec.clone() for cid in cids}

        sparsified_deltas = {}

        for cid, zero_del in zeroed_deltas.items():
            if q[cid] != D:
                mask_indices = torch.topk(delta_vec, D-q[cid], largest=False, sorted=False).indices
                zero_del[mask_indices] = 0

            temp_delta = deepcopy(list(server_deltas.values()))
            vector_to_parameters(zero_del, temp_delta)
            sparsified_deltas[cid] = {k: v for k, v in zip(keys, temp_delta)}

        return sparsified_deltas, q

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
        cgsv_vals = {}
        for cid in client_ids:
            cgsv =  compute_cgsv(server_delta, client_deltas[cid])
            self._client_wts[cid] = add_momentum_and_clip(self._client_wts[cid], cgsv, self.cfg.alpha)
            cgsv_vals[cid] = cgsv
            
        ic(cgsv_vals)
        self.res_man.log_general_metric(cgsv_vals, phase='post_agg', actor='server', metric_name='cgsv')
        # Normalize the coefficients
        self._client_wts = self.normalize_weights(self._client_wts)

        ic(self._client_wts)
        self._server_params = add_param_deltas(self._server_params, server_delta)

        if self.cfg.sparsify_gradients:
            client_deltas, q = self._sparsify_gradients(self._client_wts, server_delta, self.cfg.beta)
            self.res_man.log_general_metric(q, phase='post_agg', actor='server', metric_name='q')
            for cid in client_ids:
                self._client_params[cid] = add_param_deltas(_client_params[cid], client_deltas[cid])
                self.res_man.log_parameters(self._client_params[cid], phase='post_agg', actor=cid)

        else:
            for cid in client_ids:
                # self._client_params[cid] = add_param_deltas(_client_params[cid], server_delta)
                self._client_params[cid] = deepcopy(self._server_params)

        # Debug sparsify gradients
        try:
            self._sparsify_gradients(self._client_wts, server_delta, self.cfg.beta)
        except:
            logger.error('Error in sparsify gradients')
        # Logging code
        self.res_man.log_general_metric(self._client_wts, phase='post_agg', actor='server', metric_name='client_weights')


        self.res_man.log_parameters(self._server_params, phase='post_agg', actor='server')
        

        return CgsvOuts(server_params=self._server_params,
                        client_params=self._client_params)

