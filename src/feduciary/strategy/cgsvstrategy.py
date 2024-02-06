import logging
from collections import OrderedDict
import typing as t
from dataclasses import dataclass
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CosineSimilarity, Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector

from feduciary.config import ClientConfig, CGSVConfig
from feduciary.results.resultmanager import ResultManager
from feduciary.strategy import ABCStrategy, StrategyIns, StrategyOuts, random_client_selection, passthrough_communication
import feduciary.common.typing as fed_t
from feduciary.common.utils import generate_client_ids
from feduciary.strategy.fedoptstrategy import gradient_average_update, gradient_average_with_delta_normalize
from feduciary.strategy.fedstdevstrategy import normalize_coefficients

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

class CgsvOuts(StrategyOuts):
    client_params: fed_t.ClientParams_t

class CgsvStrategy(ABCStrategy):
    name: str = 'CgsvStrategy'

    def __init__(self, model,
                cfg: CgsvCfgProtocol,
                res_man: ResultManager):
        
        self.cfg = cfg
        client_ids = generate_client_ids(cfg.num_clients)

        if self.cfg.delta_normalize:
            self._update_fn = gradient_average_with_delta_normalize
        else:
            self._update_fn = gradient_average_update

        
        self.local_grad_norm = None
        self.server_grad_norm = None

        self._cos_sim = CosineSimilarity(dim=0)
        # NOTE: Differing in initialization here from paper as it leads to permanently zero gradients
        self._client_wts = {cid: 1.0/len(client_ids) for cid in client_ids}
        

    def receive_strategy(self, ins: fed_t.ClientResults_t) -> CgsvIns:
        return passthrough_communication(ins)

    def send_strategy(self, ids: fed_t.ClientIds_t) -> fed_t.ClientIns_t:
        '''Send custom models to respective clients'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fed_t.ClientIns(
                params=self._server_params,
                metadata={}
            )
        return clients_ins


    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> CgsvOuts:
        base_outs = CgsvOuts(
            server_params=ins.params,
            client_params=
        )
        return base_outs

    @classmethod
    def client_send_strategy(cls, ins: CgsvIns, result: fed_t.Result) -> fed_t.ClientResult:
        return fed_t.ClientResult(ins.client_params, result) 

    def train_selection(self, in_ids: fed_t.ClientIds_t, **kwargs) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    def eval_selection(self, in_ids: fed_t.ClientIds_t, **kwargs) -> fed_t.ClientIds_t:
        return super().eval_selection(in_ids, **kwargs)
    

        
    def _compute_cgsv(self, server_param, local_param):
        # self.local_grad_norm = self.server_para
        # print(server_param[0].dtype)
        # print(local_param[0].dtype)

        server_param_vec = parameters_to_vector(server_param)
        local_param_vec = parameters_to_vector(local_param)

        return self._cos_sim(server_param_vec, local_param_vec)

    def _update_coefficients(self, client_id, cgsv):
        self._client_wts[client_id] = self.cfg.alpha * self._client_wts[client_id] + (1 - self.cfg.alpha)* cgsv

        
    def _sparsify_gradients(self, client_ids):
        # TODO: Implement gradient sparsification for reward 
        pass

    def normalize_coefficients(self):
        total = 0
        for val in self._client_wts.values():
            total += val
        for key, val in self._client_wts.items():
            self._client_wts[key] = val/total

    def normalize_scalar_weights(cls, in_weights: ClientScalarWeights_t) -> ClientScalarWeights_t:
        param_keys = in_weights[list(in_weights.keys())[0]].keys()
        out_weights = {cid: {param: 0.0 for param in param_keys} for cid in in_weights.keys()}
        total_coeff = {param: 1e-9 for param in param_keys}

        for cid, coeff in in_weights.items():
            for layer, weight in coeff.items():
                total_coeff[layer] += weight

        for cid, coeff in in_weights.items():
            for layer, weight in coeff.items():
                out_weights[cid][layer] = weight/total_coeff[layer]

        return out_weights

    
    def aggregate(self, strategy_ins: AllClientIns_t) -> CgsvOuts:


        local_params = [param.data.float() for _, param in local_params_dict.items()]
        assert len(self._server_params) == len(local_params), f'Mismatch in parameter lengths'

        # print(local_params[0])
        # print(self._server_params[0].size())
        # print(self._server_params[0].norm())

        # exit(0)
        local_grads = []
        server_grads = []
        i = 0
        # print(len(self._server_params))
        for server_param, local_param in zip(self._server_params, local_params):
            i += 1
            local_delta = local_param - server_param

            norm = local_delta.norm() 
            if norm == 0:
                logger.warning(f"CLIENT [{client_id}]: Got a zero norm!")
                local_grad_norm = local_delta.mul(self.gamma)
            else:
                local_grad_norm = local_delta.div(norm).mul(self.gamma)

            weighted_local_grad = local_grad_norm.mul(self._client_wts[client_id])
            
            # server params grad is used as a buffer
            if server_param.grad is None:
                server_param.grad = weighted_local_grad
            else:
                server_param.grad.add_(weighted_local_grad)

            server_grads.append(server_param.grad.data)
            local_grads.append(local_grad_norm.data)

        cgsv = self._compute_cgsv(server_grads, local_grads)

        self._update_coefficients(client_id, cgsv)

        # Normalize the coefficients
        self.normalize_coefficients()
        # print(self._client_wts)
        # print(self._client_wts.values())
        # print(self._client_wts.keys
        return CgsvOuts(server_params=self._server_params)

