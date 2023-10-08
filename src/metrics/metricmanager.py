from collections import defaultdict
# from typing import DefaultDict
from importlib import import_module
from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .basemetric import BaseMetric
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from logging import Logger
import pandas as pd
import json

# class Updateable(object):
#     def update(self, new):
#         for key, value in new.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)

# Result for one entity to the epoch level
@dataclass
class Result:
    caller: str = ''# this is just for debugging for now
    epoch: int = -1 # epoch -1 reserved for evaluation request
    round: int = 0 # this is just for debugging for now
    size: int = 0
    metrics: dict[float] = field(default_factory=dict)

##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics, round, caller):
        self.metric_funcs = {
            name: import_module(f'.metricszoo', package=__package__).__dict__[name.title()]() for name in eval_metrics}
        self.figures = defaultdict(int) 
        self._result = Result(round=round, caller=caller)
        self._round = round

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, epoch):
        # aggregate 
        avg_metrics = {name: module.summarize() for name, module in self.metric_funcs.items()}

        avg_metrics['loss'] = self.figures['loss'] / total_len

        self._result.metrics = avg_metrics
        self._result.epoch = epoch
        self._result.size = total_len

        self.figures = defaultdict(int)

    def flush(self):
        self.figures = defaultdict(int)
        self._result = Result()
    
    @property
    def result(self):
        return self._result


def log_metric(key:str, rnd:int, metrics:dict, logger: Logger, writer:SummaryWriter):
    log_string = f'Round:{rnd} | {key.upper()} '
    for metric, stat in metrics.items():
        log_string += f'| {metric}: {stat:.4f} '
    
    logger.info(log_string)
    writer.add_scalars(key, metrics, rnd)


@dataclass
class Stats:
    mean:float = float('nan')
    maximum:float = float('nan')
    minimum:float = float('nan')
    std:float = float('nan')

# Round wise result for all participants of that round
@dataclass
class ClientResult:
    round: int = field(default=-1)
    results: dict[Result] = field(default_factory=dict)
    stats: dict[Stats] = field(default_factory=dict)
    sizes: dict[int] = field(default_factory=dict)
    
# class ResultPhase(Enum):
#     TRAIN = auto()
#     EVAL = auto()
#     EVAL_PRE_AGGREGATE = auto()


@dataclass
class AllResults:
    # This might need a default factory
    round: int = 0
    participants: dict[str] = field(default_factory=dict)
    clients_train: ClientResult = field(default_factory=ClientResult)
    clients_eval: ClientResult = field(default_factory=ClientResult)
    clients_eval_pre: ClientResult = field(default_factory=ClientResult)
    server_eval : Result = field(default_factory=Result)


class ResultManager:
    '''Accumulate and process the results'''
    full_results: list[dict] = []
    _round: int = 0

    def __init__(self, logger: Logger, writer:SummaryWriter) -> None:
        self.result = AllResults()
        self.writer = writer
        self.logger = logger

    
    def client_results(self, key:str, result:dict[Result]) -> ClientResult:
        client_result = ClientResult()
        client_result.results = result
        client_result.round = self._round
        client_result = self.compute_client_stats_and_log(key, client_result)
        return client_result
        
    def populate_stats(self, array:list) -> Stats:
        stat = Stats(-1, -1, -1, -1)
        np_array = np.array(array).astype(float)
        stat.mean = np.mean(np_array)
        stat.std = np.std(np_array)
        stat.minimum = np.min(np_array)
        stat.maximum = np.max(np_array)
        return stat

    def _round_check(self, rnd, cid):
        assert self._round == rnd, f'Mismatching rounds: {cid} :r {rnd}, global round: {self._round}'

    def compute_client_stats_and_log(self, key, client_result:ClientResult)->ClientResult:
        metrics_list=defaultdict(list)
        for cid, clnt in client_result.results.items():
            self._round_check(clnt.round, cid)
            for metric, value in clnt.metrics.items():
                metrics_list[metric].append(value)
            client_result.sizes[cid] = clnt.size

            log_metric(f'client:{cid}_{key}', self._round, clnt.metrics, self.logger, self.writer)

        for metric, vals in metrics_list.items():
            client_result.stats[metric] = self.populate_stats(vals)
            log_metric(f'{key}_{metric}', self._round, client_result.stats[metric].__dict__, self.logger, self.writer)
        
        return client_result

    
    def log_server_eval_result(self, result:Result):
        self._round_check(result.round, 'server')
        log_metric('server eval', result.round, result.metrics, self.logger, self.writer)
        self.result.server_eval = result
        self.result.participants['server_eval'] = 'server'


    def log_client_eval_result(self, result:dict[Result]):
        key = 'client_eval'
        client_result = self.client_results(key, result)
        self.result.clients_eval = client_result
        self.result.participants[key] = list(result.keys())
        return client_result

    def log_client_eval_pre_result(self, result:dict[Result]):
        key = 'client_eval_pre'
        client_result = self.client_results(key, result)
        self.result.clients_eval_pre = client_result
        self.result.participants[key] = list(result.keys())
        return client_result
    
    def log_client_train_result(self, result:dict[Result]):
        key = 'client_train'
        client_result = self.client_results(key, result)
        self.result.clients_train = client_result
        self.result.participants[key] = list(result.keys())
        return client_result
    
    def update_round_and_flush(self, rnd:int):
        # print((self.result))
        self.result.round = self._round
        self.full_results.append(asdict(self.result))
        self.result = AllResults()
        self.writer.flush()
        # print(pd.json_normalize(self.full_results))
        # print((self.full_results))
        self._round = rnd+1  # setup for the next round


    def finalize(self):
        # TODO:
        """Save results.
        """
        self.logger.info(f'[RESULTS] [Round: {self._round:03}] Save results and the global model checkpoint!')
        df = pd.json_normalize(self.full_results)
        df.to_csv('results.csv')

        self.writer.close()

        self.logger.info(f' [Round: {self._round:03}] ...finished federated learning!')


    # TBD
    # def calculate_generalization_gap(self):
    #     gen_gap = dict()
    #     curr_res: dict = self.results[self.round]
    #     for key in curr_res['clients_evaluated_out'].keys():
    #         for name in curr_res['clients_evaluated_out'][key].keys():
    #             if name in ['equal', 'weighted']:
    #                 gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
    #                 gen_gap[f'gen_gap_{key}'] = {name: gap}
    #                 self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
    #                 self.writer.flush()
    #     else:
    #         self.results[self.round]['generalization_gap'] = dict(gen_gap)
