from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
import typing as t
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import flwr.server.strategy as fl_strat
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Parameter
from torch import Tensor


from torch.optim.lr_scheduler import LRScheduler

from src.client.baseflowerclient import BaseFlowerClient
from src.config import ClientConfig, ServerConfig
from src.results.resultmanager import ResultManager

from src.client.baseclient import BaseClient, model_eval_helper
from src.metrics.metricmanager import MetricManager
from src.common.utils  import log_tqdm, log_instance, get_parameters_as_ndarray
from src.results.resultmanager import ResultManager
import src.common.typing as fed_t

from src.server.abcserver import ABCServer

from src.config import *
from src.strategy.basestrategy import BaseStrategy

# TODO: Long term todo: the server should 
#  eventually be tied directly to the server algorithm
logger = logging.getLogger(__name__)


# def get_parameters_as_ndarray(net: Module) -> list[np.ndarray]:
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]

def convert_param_dict_to_ndarray(param_dict: fed_t.ActorParams_t) -> fl.common.NDArrays:
    return [val.cpu().numpy() for val in param_dict.values()]

def create_param_dict(keys: list[str], parameters: List[np.ndarray]):
    params_itr = zip(keys, parameters)
    state_dict = OrderedDict({k: Parameter(data=Tensor(v)) for k, v in params_itr})
    return state_dict

def _nest_dict_rec(k: str, v, out: dict):
    k, *rest = k.split('.', 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v

def nest_dict(flat: dict) -> dict:
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result)
    return result

def flatten_dict(nested: dict) -> dict:
    return pd.json_normalize(nested, sep='.').to_dict('records')[0]

def flower_metrics_to_results(flwr_res: FitRes | EvaluateRes) -> fed_t.Result:
    # print(flwr_res.metrics.keys())
    nested = nest_dict(flwr_res.metrics)
    # print("Nested: ", nested)
    return fed_t.Result(actor=nested['actor'],
                        _round=nested['_round'],
                        metadata=nested['metadata'],
                        metrics=nested['metrics'],
                        size=flwr_res.num_examples)

def flower_eval_results_adapter(results: List[Tuple[ClientProxy, EvaluateRes]]) -> dict[str, fed_t.Result]:
    client_results = {}
    for client, result in results:
        client_results[client.cid] = flower_metrics_to_results(result)
    return client_results


def flower_train_results_adapter(param_keys, results: List[Tuple[ClientProxy, FitRes]]) -> fed_t.ClientResults_t:
    client_results = {}
    for client, result in results:
        nd_params = fl.common.parameters_to_ndarrays(result.parameters)
        client_params = create_param_dict(param_keys, nd_params)
        client_result = flower_metrics_to_results(result)

        client_results[client.cid] = fed_t.ClientResult1(params=client_params, result=client_result)

    return client_results


class BaseFlowerServer(ABCServer, fl_strat.Strategy):
    """Central server orchestrating the whole process of federated learning.
    """
    name: str = 'BaseFlowerServer'

    # NOTE: It is must to redefine the init function for child classes with a call to super.__init__()
    def __init__(self, 
                 clients: dict[str, BaseFlowerClient],
                 model: Module, cfg: ServerConfig,
                 strategy: BaseStrategy,
                 train_cfg: ClientConfig,
                 dataset: Dataset,
                 result_manager: ResultManager):
        self.model = model
        self.clients = clients
        self.num_clients = len(self.clients)
        self.strategy = strategy
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.server_dataset = dataset
        self.metric_manager = MetricManager(train_cfg.eval_metrics, 0, 'server')
        if result_manager:
            self.result_manager = result_manager


        # self.client_results: fed_t.ClientResults_t = defaultdict(fed_t.ClientResult1)
        # self.client_ins_dict: fed_t.ClientIns_t = defaultdict(fed_t.ClientIns)

        self.param_keys = list(self.model.state_dict().keys())
        defaults = dict(lr=train_cfg.lr)
        self._optimizer = torch.optim.Optimizer(self.model.parameters(), defaults=defaults)

        # FIXME: Maybe make instantiate calls explicit to satisfy typechecker
        lrs_partial: partial = train_cfg.lr_scheduler
        self.lr_scheduler: LRScheduler = lrs_partial(self._optimizer)

    @classmethod
    def broadcast_model(cls, client_ins: fed_t.ClientIns,
                        client: BaseFlowerClient,
                        request_type: fed_t.RequestType,
                        _round=-1) -> fed_t.RequestOutcome:
            # NOTE: Consider setting keep vars to true later if gradients are required
            client_ins.request = request_type
            client_ins._round = _round
            out = client.download(client_ins)
            return out

    def _broadcast_models(self,
                        ids: list[str],
                        clients_ins: dict[str, fed_t.ClientIns],
                        request_type=fed_t.RequestType.NULL) -> fed_t.ClientResults_t:
                        
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """

        # HACK: does lr scheduling need to be done for select ids ??
        if self.lr_scheduler:
            current_lr = self.lr_scheduler.get_last_lr()[-1]
            [client.set_lr(current_lr) for client in self.clients.values()]

        # Uncomment this when adding GPU support to server
        # self.model.to('cpu')
        results = {}

        for idx in log_tqdm(ids, desc=f'broadcasting models: ', logger=logger):
            results[idx] = self.broadcast_model(clients_ins[idx], self.clients[idx], request_type, self._round) 
        
        return results


    # FIXME: Support custom client wise ins
    def _collect_results(self,
                        ids: list[str],
                        request_type=fed_t.RequestType.NULL) -> fed_t.ClientResults_t:
                        
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """

        # Uncomment this when adding GPU support to server
        # self.model.to('cpu')
        results = {}
        for idx in log_tqdm(ids, desc=f'collecting results: ', logger=logger):
            results[idx] = self.clients[idx].upload(request_type)
        return results


    @torch.no_grad()
    def server_eval(self):
        """Evaluate the global model located at the server.
        """
        # FIXME: Formalize phase argument passing
        server_loader = DataLoader(dataset=self.server_dataset,
                                   batch_size=self.train_cfg.batch_size, shuffle=False)
        # log result
        result = model_eval_helper(self.model, server_loader, self.train_cfg, self.metric_manager, self._round)
        self.result_manager.log_general_result(result, phase='post_agg', actor='server', event='central_eval')
        return result


    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        self._round = checkpoint['round']
        # Find a way to avoid this result manager round bug repeatedly
        self.result_manager._round = checkpoint['round']

        # loss = checkpoint['loss']
    
    def save_checkpoint(self):
        torch.save({
            'round': self._round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict' : self._optimizer.state_dict(),
            }, f'server_ckpts/server_ckpt_{self._round:003}.pt')

    
    def finalize(self) -> None:
        # save checkpoint
        torch.save(self.model.state_dict(), f'final_model.pt')
        # return all_results

        
    def update(self, avlble_cids: fed_t.ClientIds_t) -> fed_t.ClientIds_t:
        # TODO: Ideally, the clients keep running their training and server should only initiate a downlink transfer request. For synced federation, the server needs to wait on clients to finish their tasks before proceeding

        """Update the global model through federated learning.
        """
        # randomly select clients

        # broadcast the current model at the server to selected clients
        train_ids = self.strategy.train_selection(in_ids=avlble_cids)
        clients_ins = self.strategy.send_strategy(train_ids)
        outcome = self._broadcast_models(train_ids, clients_ins, fed_t.RequestType.TRAIN)
        if outcome == fed_t.RequestOutcome.COMPLETE:
            train_result = self._collect_results(train_ids, fed_t.RequestType.TRAIN)
            strategy_ins = self.strategy.receive_strategy(train_result)
            strategy_outs = self.strategy.aggregate(strategy_ins)
            self.model.load_state_dict(strategy_outs.server_params)
        return train_ids

    def local_eval(self, avlble_cids: fed_t.ClientIds_t):

        eval_ids = self.strategy.eval_selection(in_ids=avlble_cids)
        eval_ins = self.strategy.send_strategy(eval_ids)
        outcome = self._broadcast_models(eval_ids, eval_ins, fed_t.RequestType.EVAL)
        if outcome == fed_t.RequestOutcome.COMPLETE:
            eval_results = self._collect_results(eval_ids, fed_t.RequestType.EVAL)

            to_log = {cid: res.result for cid, res in eval_results.items()}

            self.result_manager.log_clients_result(to_log, event='local_eval', phase='pre_agg')
        return eval_ids
    

    # FLOWER FUNCTION OVERLOADS
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_parameters_as_ndarray(self.model)
        return fl.common.ndarrays_to_parameters(ndarrays)
    
    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        # Create custom configs
        self._round = server_round-1
      
        fit_configurations = []

        client_proxies = client_manager.all()

        out_ids = self.strategy.train_selection(list(client_proxies.keys()))
        clients_ins = self.strategy.send_strategy(out_ids)

        # FIXME: dynamic client configurations
        # client_config = {"lr": 0.001}

        for out_id in out_ids:
            client = client_proxies[out_id]
            cl_in = clients_ins[out_id]

            nd_param = convert_param_dict_to_ndarray(cl_in.params)
            flower_param = fl.common.ndarrays_to_parameters(nd_param)
            cl_config = cl_in.metadata
            cl_config['_round'] = self._round

            fit_configurations.append(
                    (client, FitIns(flower_param, cl_config)) 
                )
        return fit_configurations


    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        
  
        client_results = flower_train_results_adapter(self.param_keys, results)

        strategy_ins = self.strategy.receive_strategy(client_results)
        strategy_outs = self.strategy.aggregate(strategy_ins)

        self.model.load_state_dict(strategy_outs.server_params)
        param_ndarrays = convert_param_dict_to_ndarray(strategy_outs.server_params)

        parameters_aggregated = fl.common.ndarrays_to_parameters(param_ndarrays)
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated
    
    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        
        # NOTE: Sample all is selected for now. Remember while simulating availability
        client_proxies = client_manager.all()

        out_ids = self.strategy.eval_selection(list(client_proxies.keys()))
        eval_ins = self.strategy.send_strategy(out_ids)

        eval_configurations = []

        for out_id in out_ids:
            client = client_proxies[out_id]
            cl_in = eval_ins[out_id]
            nd_param = convert_param_dict_to_ndarray(cl_in.params)
            flower_param = fl.common.ndarrays_to_parameters(nd_param)
            cl_config = cl_in.metadata
            cl_config['_round'] = self._round


            eval_configurations.append(
                    (client, EvaluateIns(flower_param, cl_config))
                )
        
        return eval_configurations


    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException]) -> Tuple[float, Dict[str, Scalar]]:
           
        client_results = flower_eval_results_adapter(results)
        clien_result_stats = self.result_manager.log_clients_result(result=client_results, phase='post_agg',event='local_eval')
        self.result_manager.flush_and_update_round(server_round-1)
        # self._round = server_round-1

        loss_agg =  clien_result_stats.stats['loss'].mean

        metrics_agg =  {k: v.mean for k, v in clien_result_stats.stats.items()}

        return loss_agg, metrics_agg # type: ignore


    def evaluate(self, server_round: int, parameters: Parameters) -> None:
        eval_res = self.server_eval()
        loss_agg =  eval_res.metrics['loss']
        # metrics_agg =  {k: v for k, v in eval_res.metrics.items()}
        metrics_agg = eval_res.metrics

        return loss_agg, metrics_agg # type: ignore
