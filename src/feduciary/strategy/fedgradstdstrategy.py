from collections import defaultdict
import typing as t
from dataclasses import dataclass, field
import numpy as np
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
# from feduciary.common.typing import ClientIns, ClientResult, Result
from feduciary.results.resultmanager import ResultManager
import feduciary.common.typing as fed_t
from feduciary.strategy.abcstrategy import *
from feduciary.strategy.basestrategy import random_client_selection, ClientInProto, weighted_parameter_averaging
from feduciary.strategy.fedoptstrategy import gradient_average_update, add_param_deltas
# from feduciary.strategy.fedstdevstrategy import gradient_average_update
### Define the configurations required for this strategy
from feduciary.common.utils import generate_client_ids
from enum import Enum
@dataclass
class FedstdevCfgProtocol(t.Protocol):
    '''Protocol for Fedstdev strategy config'''
    train_fraction: float
    eval_fraction: float
    weighting_strategy: str
    betas: list[float]
    beta_0: float
    alpha: float
    num_clients: int

class WeightingStrategy(str, Enum):
    GRAD_SIGMA_LAYER_WISE = 'grad_sigma'
    GRAD_SIGMA_SCALAR = 'grad_sigma_scalar'
    GRAD_SIGMA_BY_MU_FULL_DIM = 'grad_sigma_by_mu_full_dim'
    GRAD_SIGMA_BY_MU_SCALAR_AVG = 'grad_sigma_by_mu_scalar_avg'
    GRAD_SIGMA_BY_MU_SCALAR_WTD_AVG = 'grad_sigma_by_mu_scalar_wtd_avg'
    GRAD_SIGMA_BY_MU_LAYER_WISE = 'grad_sigma_by_mu_layer_wise'
    GRAD_SIGMA_DIRECTIONS_TRIAL = 'grad_sigma_direction_trial'
    EQUAL = 'equal'



# Type declarations
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]
Weights_T =t.TypeVar('Weights_T', ScalarWeights_t, TensorWeights_t)
# AllIns_t = dict[str, FedstdevIns]
ClientScalarWeights_t= dict[str, ScalarWeights_t]
ClientTensorWeights_t= dict[str, TensorWeights_t]

def get_dict_avg(param_dict: dict, wts: dict) -> dict:
    # Helper function to compute average of the last layer of the dictionary
    # wtd_avg = 0.0
    # wight_sums = np.sum(self.param_dims.values())
    # if wts is None:
    wts_list = list(wts.values())

    avg = np.mean(list(param_dict.values()))
    wtd_avg = np.average(list(param_dict.values()), weights=wts_list)

    return {'avg':avg, 'wtd_avg':wtd_avg}

def get_param_dict_wtd_avg(param_dict: fed_t.ActorDeltas_t, wts: dict[str, int]) -> float:
    param_list = [val.abs().mean().item() for val in param_dict.values()]
    wts_list = list(wts.values())
    wtd_avg = np.average(param_list, weights=wts_list)
    return wtd_avg # type: ignore

# def gradient_update_full_dim_weight(server_params: fed_t.ActorParams_t,client_params: fed_t.ClientParams_t, weights: ClientTensorWeights_t) -> tuple[fed_t.ActorParams_t, fed_t.ActorDeltas_t]:
#     server_deltas: fed_t.ActorDeltas_t = {}
#     for key, server_param in server_params.items():
#         for cid, client_param in client_params.items():
#             # Using FedNova Notation of delta (Δ) as (-grad ∇)
#             client_delta = client_param[key].data - server_param.data
#             server_deltas[key] = server_deltas.get(key, 0) + weights[cid] * client_delta

#     server_params = add_param_deltas(server_params, server_deltas)    
#     return server_params, server_deltas

def gradient_update_per_param(server_params: fed_t.ActorParams_t,client_params: fed_t.ClientParams_t, weights: ClientScalarWeights_t | ClientTensorWeights_t) -> tuple[fed_t.ActorParams_t, fed_t.ActorDeltas_t]:

    server_deltas: fed_t.ActorDeltas_t = {}
    for key, server_param in server_params.items():
        for cid, client_param in client_params.items():
            # Using FedNova Notation of delta (Δ) as (-grad ∇)
            client_delta = client_param[key].data - server_param.data
            server_deltas[key] = server_deltas.get(key, 0) + weights[cid][key] * client_delta
    server_params = add_param_deltas(server_params, server_deltas)    

    return server_params, server_deltas

def compute_scalar_tanh_weights(beta: float, in_weight: float) -> float:
    return 1 - np.tanh(beta*in_weight)

def add_weight_momentum(old_weights: Weights_T, omegas: Weights_T, alpha: float) -> Weights_T:
 
    # client_wt = self._client_wts[client_id]
    new_weights= {}
    # Avoiding reading and writing to a dictionary simultaneously
    for name, omega in omegas.items():
        new_weights[name] = alpha * old_weights[name] + (1 - alpha)* omega
    return new_weights # type: ignore

def add_scalar_momentum(weight: float, omega: float, alpha: float) -> float:
    return alpha * weight + (1 - alpha)* omega

def normalize_scalar_weights(in_weights: dict[str, float]) -> dict[str, float]:
    total = 1e-9
    out_weights ={}
    for val in in_weights.values():
        total += val

    for key, val in in_weights.items():
        out_weights[key] = val/total
    return out_weights


def normalize_full_dim_weights(in_weights: ClientTensorWeights_t) -> ClientTensorWeights_t:
    cid_0 = list(in_weights.keys())[0]
    param_keys = in_weights[cid_0].keys()
    client_ids = in_weights.keys()


    out_weights = {cid: {param: torch.Tensor() for param in param_keys} for cid in client_ids}

    total_coeff = {param: torch.full_like(in_weights[cid_0][param], 1e-9) for param in param_keys}

    for cid, coeff in in_weights.items():
        for layer, weight in coeff.items():
            total_coeff[layer] += weight

    for cid, coeff in in_weights.items():
        for layer, weight in coeff.items():
            out_weights[cid][layer] = weight/total_coeff[layer]
    return out_weights

def lump_tensors(in_dict: dict[str, Tensor]) -> dict[str, float]:
    '''Lumps all the tensors in the dictionary into a single value per key'''
    return {key: val.abs().mean().item() for key, val in in_dict.items()}

# Define the inputs and outputs for the strategy
@dataclass
class FedgradIns:
    client_params: fed_t.ActorParams_t
    client_grad_mus: fed_t.ActorDeltas_t
    client_grad_stds: fed_t.ActorDeltas_t

@dataclass
class FedstdevOuts:
    server_params: fed_t.ActorParams_t

AllIns_t = dict[str, FedgradIns]

class FedgradstdStrategy(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: FedstdevCfgProtocol,
                 res_man: ResultManager) -> None:
        
        # super().__init__(model, cfg)
        self.cfg = cfg
        self.res_man = res_man
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        # HACK: Strategy class should not be aware of the client ids
        client_ids = generate_client_ids(cfg.num_clients)
        self._client_ids = client_ids
        param_keys = self._server_params.keys()
        self.param_keys = list(param_keys)
        self.param_dims = {p_key: int(np.prod(list(param.size()))) for p_key, param in self._server_params.items()}

        # self._client_params: ClientParams_t = defaultdict(dict)
        self._client_ins: dict[str, float] = defaultdict()

        self.beta_dict = {param: beta for param, beta in zip(param_keys, cfg.betas)}

        # TODO: Think of ways to support dynamic client allocation
        w_0 =  1.0/len(client_ids)


        match self.cfg.weighting_strategy:
            case WeightingStrategy.GRAD_SIGMA_BY_MU_FULL_DIM:
                # tensor_0 = torch.full_like(next(iter(self._server_params.values())), w_0)
                self._client_wts = {cid: {param: torch.full_like(self._server_params[param], w_0) for param in param_keys}
                                     for cid in client_ids}
                self._client_omegas = {cid: 
                                       {param: torch.full_like(self._server_params[param], w_0) for param in param_keys}
                                       for cid in client_ids}
            case WeightingStrategy.GRAD_SIGMA_BY_MU_SCALAR_WTD_AVG | WeightingStrategy.EQUAL:
                self._client_wts = {cid: w_0 for cid in client_ids}
                self._client_omegas = {cid: w_0 for cid in client_ids}
            case WeightingStrategy.GRAD_SIGMA_BY_MU_LAYER_WISE:
                self._client_wts = {cid: {param: w_0 for param in param_keys} for cid in client_ids}
                self._client_omegas = {cid: {param: w_0 for param in param_keys} for cid in client_ids}
            case _:
                logger.error(f'Unknown weight scaling type: {self.cfg.weighting_strategy}')


        # Additional dictionaries required for this approach
        self._client_params_std: fed_t.ClientParams_t = {cid: {param: Parameter(torch.empty_like(val.data)) for param, val in self._server_params.items()} for cid in client_ids}
        self._clnt_sigma_by_mu: fed_t.ClientParams_t = {cid: {param: Parameter(torch.empty_like(val.data)) for param, val in self._server_params.items()} for cid in client_ids}
        # tracking client deltas for logging
        self._client_deltas: fed_t.ClientDeltas_t = {cid: {param: torch.empty_like(val.data) for param, val in self._server_params.items()} for cid in client_ids}


    @classmethod
    def client_receive_strategy(cls, ins: fed_t.ClientIns) -> ClientInProto:
        return  ClientInProto(in_params=ins.params)
    
    @classmethod
    def client_send_strategy(cls, ins: FedgradIns, result: fed_t.Result) -> fed_t.ClientResult:
        out_params = ins.client_params
        for key, val in ins.client_grad_mus.items():
            out_params[f'{key}_grad_mu'] = val

        for key, val in ins.client_grad_stds.items():
            out_params[f'{key}_grad_std'] = val

        return fed_t.ClientResult(params=out_params, result=result) 

    def receive_strategy(self, results: fed_t.ClientResults_t) -> AllIns_t:
        strat_ins = {}
        for cid, res in results.items():
            client_params= {}
            client_grad_mus = {}
            client_grad_stds = {}

            for param in self.param_keys:
                client_params[param] = res.params[param]
                client_grad_mus[param] = res.params[f'{param}_grad_mu']
                client_grad_stds[param] = res.params[f'{param}_grad_std']


            strat_ins[cid] = FedgradIns(
                client_params=client_params,
                client_grad_mus=client_grad_mus,
                client_grad_stds=client_grad_stds) 
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

    @classmethod
    def layer_wise_sbm_aggregate(cls,
                                 client_wts,
                                 client_grad_mus, client_grad_stds, client_ids, alpha, beta_dict) -> tuple[dict, dict, dict]:
        client_omegas = {}
        clnt_sigma_by_mu = {}
        for cid in client_ids:
            clnt_sigma_by_mu[cid] = cls._compute_sigma_by_mu_full(client_grad_stds[cid], client_grad_mus[cid])
            omega = cls._compute_layer_wise_scaled_weights(beta_dict, clnt_sigma_by_mu[cid])
            client_wts[cid] = add_weight_momentum(
                client_wts[cid], omega, alpha)
            client_omegas[cid] = omega
        client_wts = cls.normalize_layer_wise_weights(client_wts)

        return client_wts, client_omegas, clnt_sigma_by_mu

    @classmethod
    def full_dim_sbm_aggregate(cls,
                                 client_wts,
                                 client_grad_mus, client_grad_stds, client_ids, alpha, beta_0) -> tuple[dict, dict, dict]:
        client_omegas = {}
        clnt_sigma_by_mu = {}
        for cid in client_ids:
            clnt_sigma_by_mu[cid] = cls._compute_sigma_by_mu_full(client_grad_stds[cid], client_grad_mus[cid])

            omega = cls._compute_full_dim_weights(beta_0, clnt_sigma_by_mu[cid])

            client_wts[cid] = add_weight_momentum(
                client_wts[cid], omega, alpha)
            client_omegas[cid] = omega
        client_wts = normalize_full_dim_weights(client_wts)

        return client_wts, client_omegas, clnt_sigma_by_mu  
     
    def layer_wise_sbm_log(self, client_ids,
                           client_wts: ClientScalarWeights_t,
                           client_omegas: ClientScalarWeights_t,
                           clnt_sigma_by_mu: dict[str, fed_t.ActorDeltas_t]):

        extended_omegas = {}
        extended_wts = {}
        ext_grad_sbm = {}
        for cid in client_ids:
            # logging omega and weight average
            avg_omegas = get_dict_avg(client_omegas[cid], self.param_dims)
            avg_wts = get_dict_avg(client_wts[cid], self.param_dims)
            lumped_sbm = lump_tensors(clnt_sigma_by_mu[cid])
            ext_grad_sbm[cid] = lumped_sbm | get_dict_avg(lumped_sbm, self.param_dims)

            copied_omegas = client_omegas[cid].copy()
            extended_omegas[cid] = copied_omegas | avg_omegas

            copied_wts = client_wts[cid].copy()
            extended_wts[cid] = copied_wts | avg_wts

        self.res_man.log_general_metric(extended_omegas, f'omegas', 'server', 'post_agg')
        self.res_man.log_general_metric(extended_wts, metric_name=f'client_weights/', phase='post_agg', actor='server')
        self.res_man.log_general_metric(ext_grad_sbm, f'grad_sigma_by_mu', 'server', 'post_agg')


    def layer_wise_custom_aggregate(self,
                                 client_wts,
                                 client_grad_mus, client_grad_stds, client_ids, alpha, beta_dict) -> tuple[dict, dict, dict]:
        client_omegas = {}
        clnt_sigma_by_mu = {}
        for cid in client_ids:
            
            clnt_sigma_by_mu[cid] = self._compute_sigma_by_mu_full(client_grad_stds[cid], client_grad_mus[cid])


            omega = {}
            for key, val in clnt_sigma_by_mu.items():
                sigma_scaled = beta_dict[key]*val.data
                tanh_std = 1 - torch.tanh(sigma_scaled)
                omega[key] = tanh_std.mean().item()

            client_wts[cid] = add_weight_momentum(
                client_wts[cid], omega, alpha)
            client_omegas[cid] = omega
        client_wts = self.normalize_layer_wise_weights(client_wts)

        # self.res_man.log_general_metric(_________, f'grad_sigma_by_mu', 'server', 'post_agg')

        return client_wts, client_omegas, clnt_sigma_by_mu

    def scalar_sbm_aggregate(self, client_wts: dict[str, float],
                                 client_grad_mus,
                                 client_grad_stds,
                                 client_ids,
                                 alpha, beta_0) -> tuple[dict, dict, dict]:
        client_omegas = {}
        clnt_sigma_by_mu = {}
        lumped_sbm = {}
        for cid in client_ids:
            clnt_sigma_by_mu[cid] = self._compute_sigma_by_mu_full(client_grad_stds[cid], client_grad_mus[cid])
            lumped_sigma_by_mu = get_param_dict_wtd_avg(clnt_sigma_by_mu[cid], self.param_dims)
            lumped_omega = compute_scalar_tanh_weights(beta_0, lumped_sigma_by_mu)

            client_wts[cid] = add_scalar_momentum(
                client_wts[cid], lumped_omega, alpha)
            lumped_sbm[cid] = lumped_sigma_by_mu  
            client_omegas[cid] = lumped_omega
        client_wts = normalize_scalar_weights(client_wts)
        return client_wts, client_omegas, lumped_sbm
    
    def scalar_sbm_log(self, client_wts, client_omegas, clnt_sigma_by_mu):
        self.res_man.log_general_metric(client_omegas, f'omegas', 'server', 'post_agg')
        self.res_man.log_general_metric(client_wts, metric_name=f'client_weights', phase='post_agg', actor='server')
        self.res_man.log_general_metric(clnt_sigma_by_mu, f'grad_sigma_by_mu', 'server', 'post_agg')


   
    def _log_grads(self, client_grad_mus, client_grad_stds, client_ids):
        for cid in client_ids:
            self.res_man.log_parameters(client_grad_mus[cid], f'grad_mus/{cid}', 'server', 'post_agg')
            self.res_man.log_parameters(client_grad_stds[cid], f'grad_stds/{cid}', 'server', 'post_agg')

    def aggregate(self, strategy_ins: AllIns_t) -> FedstdevOuts:
        # calculate client weights according to sample sizes
  
        clients_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        client_grad_mus = {cid: inp.client_grad_mus for cid, inp in strategy_ins.items()}
        client_grad_stds = {cid: inp.client_grad_stds for cid, inp in strategy_ins.items()}
        # ic(_clients_params_std.keys())
        client_ids = list(clients_params.keys())

        # ic(_client_ids)
        match self.cfg.weighting_strategy:
            case WeightingStrategy.GRAD_SIGMA_SCALAR:
                pass
            case WeightingStrategy.GRAD_SIGMA_LAYER_WISE:
                pass
            case WeightingStrategy.GRAD_SIGMA_BY_MU_FULL_DIM:
                self._client_wts, self._client_omegas, self._client_sigma_by_mu = self.full_dim_sbm_aggregate(self._client_wts, client_grad_mus, client_grad_stds, client_ids, self.cfg.alpha, self.beta_dict)
                
                lump_wts = {cid: lump_tensors(wts) for cid, wts in self._client_wts.items()}
                lump_omegas = {cid: lump_tensors(omg) for cid, omg in self._client_omegas.items()}
                self.layer_wise_sbm_log(self._client_ids,
                                        lump_wts,
                                        lump_omegas, self._client_sigma_by_mu)
                self._server_params, server_deltas = gradient_update_per_param(self._server_params, clients_params, self._client_wts)
            case WeightingStrategy.GRAD_SIGMA_BY_MU_SCALAR_AVG:
                pass
            case WeightingStrategy.GRAD_SIGMA_BY_MU_SCALAR_WTD_AVG:
                self._client_wts, self._client_omegas, self._client_sigma_by_mu = self.scalar_sbm_aggregate(self._client_wts, client_grad_mus, client_grad_stds, client_ids, self.cfg.alpha, self.cfg.beta_0)
                self.scalar_sbm_log(self._client_wts, self._client_omegas, self._client_sigma_by_mu)
                self._server_params, server_deltas = gradient_average_update(self._server_params, clients_params, self._client_wts)

            case WeightingStrategy.GRAD_SIGMA_BY_MU_LAYER_WISE:
                self._client_wts, self._client_omegas, self._client_sigma_by_mu = self.layer_wise_sbm_aggregate(self._client_wts, client_grad_mus, client_grad_stds, client_ids, self.cfg.alpha, self.beta_dict)
                self.layer_wise_sbm_log(self._client_ids,
                                        self._client_wts, self._client_omegas, self._client_sigma_by_mu)
                self._server_params, server_deltas = gradient_update_per_param(self._server_params, clients_params, self._client_wts)
                
            case WeightingStrategy.GRAD_SIGMA_DIRECTIONS_TRIAL:
                self._client_wts, self._client_omegas, self._client_sigma_by_mu = self.layer_wise_custom_aggregate(self._client_wts, client_grad_mus, client_grad_stds, client_ids, self.cfg.alpha, self.beta_dict)
                self.layer_wise_sbm_log(self._client_ids,
                                        self._client_wts,
                                        self._client_omegas, self._client_sigma_by_mu)
                self._server_params, server_deltas = gradient_update_per_param(self._server_params, clients_params, self._client_wts)
            case WeightingStrategy.EQUAL:

                self._server_params = weighted_parameter_averaging(self._server_params, clients_params, self._client_wts)
                self.res_man.log_general_metric(self._client_wts, metric_name=f'client_weights', phase='post_agg', actor='server')
            case _:
                logger.error(f'Unknown weight scaling type: {self.cfg.weighting_strategy}')
                raise ValueError(f'Unknown weight scaling type: {self.cfg.weighting_strategy}')


        self._log_grads(client_grad_mus, client_grad_stds, client_ids)


        outs = FedstdevOuts(server_params=self._server_params)

        return outs
    

    def _common_log_call(self, extended_omegas, ext_grad_mus, ext_grad_sigmas, extended_wts, cid):
        # LOGGING CODE
        # Not in use currently
        self.res_man.log_general_metric(extended_omegas, f'omegas/{cid}', 'server', 'post_agg')

        self.res_man.log_general_metric(ext_grad_mus, f'grad_mus/{cid}', 'server', 'post_agg')

        self.res_man.log_general_metric(ext_grad_sigmas, f'grad_sigmas/{cid}', 'server', 'post_agg')

        self.res_man.log_general_metric(extended_wts, metric_name=f'client_weights/{cid}', phase='post_agg', actor='server')
    
    @classmethod
    def _compute_sigma_by_mu_full(cls, sigmas: fed_t.ActorDeltas_t, mus: fed_t.ActorDeltas_t) -> fed_t.ActorDeltas_t:
        '''Computes sigma/mu for each parameter'''
        # assert(sigmas.keys() == mus.keys())
        sigma_by_mu = {}
        for (key, sigma), mu in zip (sigmas.items(), mus.values()):
            mask = (mu.data!=0)
            sbm = sigma.clone()

            sbm[mask] = sigma[mask]/ mu[mask].abs()

            sigma_by_mu[key] = sbm
        return sigma_by_mu
    
    @classmethod
    def _compute_sigma_by_mu_layer_wise(cls, sigmas: fed_t.ActorDeltas_t, mus: fed_t.ActorDeltas_t) -> dict[str, float]:
        '''Computes sigma/mu for each layer'''
        sigmas_layer_wise = lump_tensors(sigmas)
        mus_layer_wise = lump_tensors(mus)
        sigma_by_mu = {}
        for (key, sigma), mu in zip (sigmas_layer_wise.items(), mus_layer_wise.values()):
            sigma_by_mu[key] = sigma/mu
        return sigma_by_mu
    
    @classmethod
    def _compute_sigma_lumped_by_mu_lumped(cls, sigmas: fed_t.ActorDeltas_t, mus: fed_t.ActorDeltas_t, param_dims: dict[str, float]) -> float:
        '''Computes sigma/mu as wtd average of sigma by wtd average of mu'''
        # assert(sigmas.keys() == mus.keys())
        sigmas_layer_wise = lump_tensors(sigmas)
        mus_layer_wise = lump_tensors(mus)
        sigma_wtd = get_dict_avg(sigmas_layer_wise, param_dims)
        mu_wtd = get_dict_avg(mus_layer_wise, param_dims)
        return sigma_wtd['wtd_avg']/mu_wtd['wtd_avg']

    @classmethod
    def normalize_layer_wise_weights(cls, in_weights: ClientScalarWeights_t) -> ClientScalarWeights_t:
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
    def _compute_layer_wise_scaled_weights(cls, betas: dict[str, float], std_dict: dict[str, Parameter]) -> dict[str, float]:
        weights = {}
        for key, val in std_dict.items():
            sigma_scaled = betas[key]*val.data
            tanh_std = 1 - torch.tanh(sigma_scaled)
            weights[key] = tanh_std.mean().item()
        return weights
        
    
    @classmethod
    def _compute_full_dim_weights(cls, betas: dict[str, float], std_dict: dict[str, Parameter]) -> dict[str, Tensor]:
        weights = {}
        for key, val in std_dict.items():
            sigma_scaled = betas[key]*val.data
            tanh_std = 1 - torch.tanh(sigma_scaled)
            weights[key] = tanh_std
        return weights


    # @classmethod
    # def _compute_lumped_weight(cls, betas: dict[str, float], param_dims: dict[str, int], std_dict: dict[str, Parameter]) -> float:
    #     weights = cls._compute_layer_wise_scaled_weights(betas, std_dict)
    #     lumped_weight = get_dict_avg(weights, param_dims)
    #     return lumped_weight['wtd_avg']