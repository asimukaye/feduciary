from collections import defaultdict
import typing as t
from dataclasses import dataclass, field
import numpy as np
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
from feduciary.common.typing import ClientIns, ClientResult1, Result 
import feduciary.common.typing as fed_t
from feduciary.strategy.abcstrategy import *
from feduciary.strategy.basestrategy import random_client_selection
from feduciary.strategy.fedoptstrategy import gradient_average_update
### Define the configurations required for this strategy
from feduciary.common.utils import generate_client_ids

@dataclass
class FedstdevCfgProtocol(t.Protocol):
    '''Protocol for Fedstdev strategy config'''
    train_fraction: float
    eval_fraction: float
    weighting_strategy: str
    betas: list[float]
    alpha: float
    num_clients: int


# Type declarations
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]
Weights_T =t.TypeVar('Weights_T', ScalarWeights_t, TensorWeights_t)
# AllIns_t = dict[str, FedstdevIns]
ClientScalarWeights_t= dict[str, ScalarWeights_t]
ClientTensorWeights_t= dict[str, TensorWeights_t]

def gradient_update_per_param(server_params: fed_t.ActorParams_t,client_params: fed_t.ClientParams_t,
    weights: ClientScalarWeights_t) -> tuple[fed_t.ActorParams_t, fed_t.ActorDeltas_t]:

    server_deltas: fed_t.ActorDeltas_t = {}
    
    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            client_delta = client_param[key].data - server_param.data

            server_deltas[key] = server_deltas.get(key, 0) + weights[cid][key] * client_delta

    for key, delta in server_deltas.items():
        server_params[key].data.add_(delta)
    
    return server_params, server_deltas

def add_weight_momentum(old_weights: Weights_T, omegas: Weights_T, alpha: float) -> Weights_T:
 
    # client_wt = self._client_wts[client_id]
    new_weights= {}
    # Avoiding reading and writing to a dictionary simultaneously
    for name, omega in omegas.items():
        new_weights[name] = alpha * old_weights[name] + (1 - alpha)* omega
    return new_weights # type: ignore

        
class FedstdevStrategy(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: FedstdevCfgProtocol) -> None:
        
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        # HACK: Strategy class should not be aware of the client ids
        client_ids = generate_client_ids(cfg.num_clients)
        param_keys = self._server_params.keys()
        self.param_keys = list(param_keys)
        self.param_dims = {p_key: np.prod(list(param.size())) for p_key, param in self._server_params.items()}

        # self._client_params: ClientParams_t = defaultdict(dict)
        self._client_ins: dict[str, float] = defaultdict()

        self.beta_dict = {param: beta for param, beta in zip(param_keys, cfg.betas)}

        # TODO: Think of ways to support dynamic client allocation
        per_client_per_param_imp =  1.0/len(client_ids)
        self._client_wts: ClientScalarWeights_t = {cid: {param: per_client_per_param_imp for param in param_keys} for cid in client_ids}


        self._client_omegas: ClientScalarWeights_t = {cid: {param: per_client_per_param_imp for param in param_keys} for cid in client_ids}


        # Additional dictionaries required for this approach
        self._client_params_std: fed_t.ClientParams_t = {cid: {param: Parameter(torch.empty_like(val.data)) for param, val in self._server_params.items()} for cid in client_ids}
        self._clnt_sigma_by_mu: fed_t.ClientParams_t = {cid: {param: Parameter(torch.empty_like(val.data)) for param, val in self._server_params.items()} for cid in client_ids}
        # tracking client deltas for logging
        self._client_deltas: fed_t.ClientDeltas_t = {cid: {param: torch.empty_like(val.data) for param, val in self._server_params.items()} for cid in client_ids}


    # Define the inputs and outputs for the strategy
    @dataclass
    class FedstdevIns:
        client_params: fed_t.ActorParams_t
        client_param_stds: fed_t.ActorParams_t

    @dataclass
    class FedstdevInsProtocol(t.Protocol):
        client_params: fed_t.ActorParams_t
        client_param_stds: fed_t.ActorParams_t

    @dataclass
    class FedstdevOuts:
        server_params: fed_t.ActorParams_t
    
    AllIns_t = dict[str, FedstdevIns]

    @classmethod
    def client_receive_strategy(cls, ins: ClientIns) -> FedstdevOuts:
        base_outs = cls.FedstdevOuts(
            server_params=ins.params,
        )
        return base_outs
    
    @classmethod
    def client_send_strategy(cls, ins: FedstdevInsProtocol, result: Result) -> ClientResult1:
        out_params = ins.client_params
        for key, val in ins.client_param_stds.items():
            out_params[f'{key}_std'] = val

        return ClientResult1(params=out_params, result=result) 

    def receive_strategy(self, results: fed_t.ClientResults_t) -> AllIns_t:
        client_params= {}
        client_stdev = {}
        strat_ins = {}
        for cid, res in results.items():
            for param in self.param_keys:
                client_params[param] = res.params[param]
                client_stdev[param] = res.params[f'{param}_std']

            strat_ins[cid] = self.FedstdevIns(client_params=client_params, client_param_stds=client_stdev) 
        return strat_ins
    
    def send_strategy(self, ids: fed_t.ClientIds_t) -> fed_t.ClientIns_t:
        '''Simple send the same model to all clients strategy'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = ClientIns(
                params=self._server_params,
                metadata={}
            )
        return clients_ins
    
    def train_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)

    def aggregate(self, strategy_ins: AllIns_t) -> FedstdevOuts:
        # calculate client weights according to sample sizes
  
        _clients_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        _clients_params_std = {cid: inp.client_param_stds for cid, inp in strategy_ins.items()}
        # ic(_clients_params_std.keys())
        _client_ids = list(_clients_params.keys())
        # ic(_client_ids)


        if self.cfg.weighting_strategy == 'tanh':
            for cid in _client_ids:
                omega = self._compute_scaled_weights(self.beta_dict, _clients_params_std[cid])

                self._client_wts[cid] = add_weight_momentum(self._client_wts[cid], omega, self.cfg.alpha)

                self._client_omegas[cid] = omega
        elif self.cfg.weighting_strategy =='tanh_sigma_by_mu':
            for cid in _client_ids:
                self._clnt_sigma_by_mu[cid] = self._compute_sigma_by_mu(_clients_params_std[cid], _clients_params[cid])

                omega = self._compute_scaled_weights(self.beta_dict, self._clnt_sigma_by_mu[cid])

                self._client_wts[cid] = add_weight_momentum(self._client_wts[cid], omega, self.cfg.alpha)

                self._client_omegas[cid] = omega
        else:
            logger.error(f'Unknown weight scaling type: {self.cfg.weighting_strategy}')

        self._client_wts = self.normalize_scalar_weights(self._client_wts)

        self._server_params, server_deltas = gradient_update_per_param(self._server_params, _clients_params, self._client_wts)

        outs = self.FedstdevOuts(server_params=self._server_params)

        return outs
    
    @classmethod
    def _compute_sigma_by_mu(cls, sigmas: fed_t.ActorParams_t, mus: fed_t.ActorParams_t) -> fed_t.ActorParams_t:
        '''Computes sigma/mu for each parameter'''
        assert(sigmas.keys() == mus.keys())
        sigma_by_mu = {}
        for (key, sigma), mu in zip (sigmas.items(), mus.values()):
            mask = (mu.data!=0)
            sbm = sigma.clone()

            sbm[mask] = sigma[mask]/ mu[mask].abs()

            sigma_by_mu[key] = sbm
        return sigma_by_mu

    @classmethod
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
        
    @classmethod
    def _compute_scaled_weights(cls, betas: dict[str, float], std_dict: dict[str, Parameter]) -> dict[str, float]:
        weights = {}
        for key, val in std_dict.items():
            # # HACK to support Tensors and parameters both
            # if isinstance(val, Parameter):
            #     val = val.data
            sigma_scaled = betas[key]*val.data
            tanh_std = 1 - torch.tanh(sigma_scaled)
            weights[key] = tanh_std.mean().item()
        return weights
    
