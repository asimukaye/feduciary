
from collections import defaultdict
from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from logging import Logger
import pandas as pd
import wandb
from copy import deepcopy
from src.config import *

# Result for one entity to the epoch level
@dataclass
class Result:
    caller: str = ''# this is just for debugging for now
    epoch: int = -1 # epoch -1 reserved for evaluation request
    round: int = 0 # this is just for debugging for now
    size: int = 0  # dataset size used to generate this result object
    metrics: dict[float] = field(default_factory=dict)

def scrub(obj, bad_key):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == bad_key:
                del obj[key]
            else:
                scrub(obj[key], bad_key)
    
def log_metric(key:str, rnd:int, metrics:dict, logger: Logger, writer:SummaryWriter):
    log_string = f'Round:{rnd} | {key.upper()} '
    for metric, stat in metrics.items():
        log_string += f'| {metric}: {stat:.4f} '
    logger.debug(log_string)


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
    results: dict[str, Result] = field(default_factory=dict)
    stats: dict[str,Stats] = field(default_factory=dict)
    sizes: dict[str,int] = field(default_factory=dict)
    
@dataclass
class AllResults:
    # This might need a default factory
    round: int = 0
    participants: dict[str] = field(default_factory=dict)
    clients_train: ClientResult = field(default_factory=ClientResult)
    clients_eval: ClientResult = field(default_factory=ClientResult)
    clients_eval_pre: ClientResult = field(default_factory=ClientResult)
    # client
    server_eval : Result = field(default_factory=Result)

@dataclass
class ClientParameters:
    round: int
    params: dict[str, Tensor]


# TODO: Add nesting later to improve the 

# @dataclass
# class ClientStats
class ResultManager:
    '''Accumulate and process the results'''
    # Potentially runaway list commented
    # full_results: list[dict] = []
    _round: int = 0

    def __init__(self, cfg:SimConfig, logger: Logger) -> None:
        self.result = AllResults()
        self.last_result = AllResults()

        self.writer = SummaryWriter()
        self.logger = logger
        self.cfg = cfg

    
    def get_client_results(self, key:str, result:dict[str, Result]) -> ClientResult:
        client_result = ClientResult()
        client_result.results = result
        client_result.round = self._round
        client_result = self.compute_client_stats_and_log(key, client_result)
        return client_result

    @staticmethod    
    def compute_stats(array:list) -> Stats:
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
        # Maybe per client logging could be disabled later
        for cid, clnt in client_result.results.items():
            self._round_check(clnt.round, cid)
            for metric, value in clnt.metrics.items():
                metrics_list[metric].append(value)
            client_result.sizes[cid] = clnt.size

            log_metric(f'client:{cid}_{key}', self._round, clnt.metrics, self.logger, self.writer)

        for metric, vals in metrics_list.items():
            client_result.stats[metric] = self.compute_stats(vals)
            log_metric(f'{key}_{metric}', self._round, client_result.stats[metric].__dict__, self.logger, self.writer)
        
        return client_result

    
    def log_server_eval_result(self, result:Result):
        self._round_check(result.round, 'server')
        log_metric('server_eval', result.round, result.metrics, self.logger, self.writer)
        self.result.server_eval = result
        self.result.participants['server_eval'] = 'server'


    def log_client_eval_result(self, result:dict[Result]):
        key = 'client_eval'
        client_result = self.get_client_results(key, result)
        self.result.clients_eval = client_result
        # self.logger.debug(f'Participants in {key}: {result.keys()}')
        self.result.participants[key] = list(result.keys())
        return client_result

    def log_client_eval_pre_result(self, result:dict[Result]):
        key = 'client_eval_pre'
        client_result = self.get_client_results(key, result)
        self.result.clients_eval_pre = client_result
        self.result.participants[key] = list(result.keys())
        # self.logger.debug(f'Participants in {key}: {result.keys()}')
        return client_result
    
    def log_client_train_result(self, result:dict[Result]):
        key = 'client_train'
        client_result = self.get_client_results(key, result)
        self.result.clients_train = client_result
        self.result.participants[key] = list(result.keys())
        # self.logger.debug(f'Participants in {key}: {result.keys()}')
        return client_result
    

    def update_round_and_flush(self, rnd:int):
        self.result.round = self._round
        self.last_result = deepcopy(self.result)
        self.save_results(asdict(self.result))
        self.result = AllResults()
        self.writer.flush()

        self._round = rnd+1  # setup for the next round

    def log_wandb_and_tb(self, result_dict: dict, tag='results'):

        wandb_dict = pd.json_normalize(result_dict, sep='/').to_dict('records')[0]

        if self.cfg.use_tensorboard:
            self.writer.add_scalars(tag, wandb_dict, global_step=self._round)

        if self.cfg.use_wandb:
            wandb.log(wandb_dict)

    def save_as_csv(self, result_dict: dict, filename='results.csv'):
        df = pd.json_normalize(result_dict)
        if self._round == 0:
            df.to_csv(filename, mode='w', index=False, header=True)
        else:
            df.to_csv(filename, mode='a', index=False, header=False)

    def scrub_dictionary(self, in_dict: dict, remove_keys: list):
        scrubbed_dict = deepcopy(in_dict)
        for key in remove_keys:
            scrub(scrubbed_dict, key)
        if 'round' in remove_keys:
            scrubbed_dict['round'] = self._round
        return scrubbed_dict

    def save_results(self, result_dict: dict):
        remove_keys = ['caller', 'epoch', 'sizes',
                        'size', 'participants', 'round']
        cleaned_dict = self.scrub_dictionary(result_dict, remove_keys)

        self.log_wandb_and_tb(cleaned_dict)
        self.save_as_csv(result_dict=result_dict)



    def finalize(self):
        # TODO:
        """Save results.
        """
        self.logger.debug(f'[RESULTS] [Round: {self._round:03}] Save results and the global model checkpoint!')
        # df = pd.json_normalize(self.full_results)
        # df.to_csv('results.csv')
        self.writer.close()

        self.logger.info(f' [Round: {self._round:03}] ...finished federated learning!')
        return self.last_result


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



# class Updateable(object):
#     def update(self, new):
#         for key, value in new.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)

