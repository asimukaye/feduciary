
from collections import defaultdict
from torch import Tensor
from torch.nn import Parameter
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
    actor: str = ''# this is just for debugging for now
    epoch: int = -1 # epoch -1 reserved for evaluation request
    round: int = 0 # this is just for debugging for now
    size: int = 0  # dataset size used to generate this result object
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

@dataclass
class Stats:
    mean: float = float('nan')
    maximum: float = float('nan')
    minimum: float = float('nan')
    std: float = float('nan')

# Round wise result for all participants of that round
@dataclass
class ClientResult:
    round: int = field(default=-1)
    results: dict[str, Result] = field(default_factory=dict)
    stats: dict[str, Stats] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)

@dataclass
class EventResult:
    round: int

@dataclass
class ClientParameters:
    round: int
    params: dict[str, Tensor]

# Remove a key in a nested dictionary
def scrub_key(obj, bad_key):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == bad_key:
                del obj[key]
            else:
                scrub_key(obj[key], bad_key)
    
def log_metric(key:str, rnd: int, metrics:dict, logger: Logger):
    log_string = f'Round:{rnd} | {key.upper()} '
    for metric, stat in metrics.items():
        log_string += f'| {metric}: {stat:.4f} '
    logger.debug(log_string)

# def rearrange_dict_per_client(main_dict: dict[int, dict[str, np.ndarray]]):
#     out_dict = defaultdict(dict)   
#     for round, in_dict in main_dict.items():
#         for clnt, feat_dict in in_dict.items():
#             out_dict[clnt][round] = feat_dict
#     return out_dict


# TODO: Add plotting function for wandb
# 
class ResultManager:
    '''Accumulate and process the results'''
    _round: int = 0

    def __init__(self, cfg: SimConfig, logger: Logger) -> None:
        # Event actor metric form
        self.result_dict = defaultdict()
        self.last_result = defaultdict()

        # Metric Event Actor form
        self.metric_event_actor_dict = defaultdict(lambda: defaultdict(dict))

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
    def compute_stats(array: list) -> Stats:
        stat = Stats(-1, -1, -1, -1)
        np_array = np.array(array).astype(float)
        stat.mean = np.mean(np_array)
        stat.std = np.std(np_array)
        stat.minimum = np.min(np_array)
        stat.maximum = np.max(np_array)
        return stat

    def _round_check(self, rnd, cid):
        assert self._round == rnd, f'Mismatching rounds: {cid} :r {rnd}, global round: {self._round}'

    def compute_client_stats_and_log(self, event: str, client_result: ClientResult) -> ClientResult:

        metrics_list=defaultdict(list)
        dict_by_metric = defaultdict(dict)
        # Maybe per client logging could be disabled later
        for cid, clnt_result in client_result.results.items():
            self._round_check(clnt_result.round, cid)

            for metric, value in clnt_result.metrics.items():
                metrics_list[metric].append(value)
                dict_by_metric[metric][cid] = value

            dict_by_metric['size'][cid] = clnt_result.size
            client_result.sizes[cid] = clnt_result.size

            log_metric(f'client:{cid}_{event}', self._round, clnt_result.metrics, self.logger)

        for metric, vals in metrics_list.items():
            # TODO: vals can be replaced by dict_by_metri.values()

            client_result.stats[metric] = self.compute_stats(vals)
            log_metric(f'{event}_{metric}', self._round, client_result.stats[metric].__dict__, self.logger)
        
        for metric, stat in client_result.stats.items():
            for stat_name, value in asdict(stat).items():
                dict_by_metric[metric][stat_name] = value

        for metric in dict_by_metric.keys():
            self.metric_event_actor_dict[metric][event] = dict_by_metric[metric]
        return client_result
    
    def log_server_eval_result(self, result: Result):
        self._round_check(result.round, 'server')
        log_metric('server_eval', result.round, result.metrics, self.logger)
        self.result_dict['server_eval'] = asdict(result)
        for metric, value in result.metrics.items():
            self.metric_event_actor_dict[metric]['central_eval']['server'] = value
        self.metric_event_actor_dict['size']['central_eval']['server'] =  result.size
        

    def log_client_result(self, result: dict[Result], event='client_eval'):
        client_result = self.get_client_results(event, result)
        self.result_dict[event] = asdict(client_result)
        # self.result.participants[key] = list(result.keys())
        return client_result

    def log_parameters(self, model_params: dict[str, Parameter], event: str, actor: str):
        out_dict = {}
        avg = 0.0
        weighted_avg = 0.0
        layer_dims = 0
        for name, param in model_params.items():
            param_detached = param.detach().cpu()
            out_dict[name]= param.abs().mean().item()
            avg += out_dict[name]
            layer_dim = np.prod(param_detached.size())
            weighted_avg += out_dict[name]*layer_dim
            layer_dims += layer_dim


        avg = avg/len(model_params)
        weighted_avg = weighted_avg/layer_dims

        out_dict['avg'] = avg
        out_dict['wtd_avg'] = weighted_avg

        # self.result_dict[event] = out_dict
        for k, val in out_dict.items():
            self.metric_event_actor_dict[f'param/{k}'][event][actor] = val

        # return out_dict

    def log_general_metric(self, metric, metric_name: str, event: str, actor: str):
        if isinstance(metric, (dict, float, int)):
            self.metric_event_actor_dict[metric_name][event][actor] = metric
        else:
            logger.error(f'Metric logging of type: {type(metric)} is not supported')
            raise TypeError

    def update_round_and_flush(self, rnd:int):
        self.result_dict['round'] = self._round
        self.metric_event_actor_dict['round'] = [self._round]

        self.last_result = deepcopy(self.result_dict)
        self.save_results(self.result_dict)
        self.result_dict = defaultdict()
        self._round = rnd+1  # setup for the next round

    def log_wandb_and_tb(self, result_dict: dict, tag='results'):
        # TODO: Reorg dictionary for multi-line plotting

        remove_keys = ['actor', 'epoch', 'sizes',
                        'size', 'round']
        # scrubbed_dict = self.scrub_dictionary(result_dict, remove_keys)

        scrubbed_dict = self.scrub_dictionary(self.metric_event_actor_dict, remove_keys)

        # Fully flatten the results
        # flattened_dict = pd.json_normalize(scrubbed_dict, sep='/').to_dict('records')[0]

        flattened_dict = pd.json_normalize(scrubbed_dict, sep='/').to_dict('records')[0]

        if self.cfg.use_tensorboard:
            self.writer.add_scalars(tag, flattened_dict, global_step=self._round)
            self.writer.flush()
        
        # ic(flattened_dict)

        if self.cfg.use_wandb:
            # wandb_dict = self.wandb_plots()
            wandb.log(flattened_dict)
            # wandb.log(wandb_dict)


    def wandb_plots(self, plot_keys: list =None):
        # What a waste of time this was...
        wandb_dict = {}
        # in_dict = defaultdict(dict)
        in_dict = pd.json_normalize(self.metric_event_actor_dict, sep='/', max_level=1).to_dict('records')[0]

        scrub_key(in_dict, 'std')
        plot_keys = list(in_dict.keys())
        plot_keys.remove('round')
        # ic(plot_keys)
    
        for plot_key in plot_keys:
            # Convert each item to a list
            for key in in_dict[plot_key].keys():
                in_dict[plot_key][key] = [in_dict[plot_key][key]]
            wandb_dict[plot_key] =  wandb.plot.line_series(in_dict['round'], list(in_dict[plot_key].values()), keys=list(in_dict[plot_key].keys()), title=plot_key, xname='round')
        return wandb_dict    


    def save_as_csv(self, result_dict: dict, filename='results.csv'):
        df = pd.json_normalize(result_dict)
        if self._round == 0:
            df.to_csv(filename, mode='w', index=False, header=True)
        else:
            df.to_csv(filename, mode='a', index=False, header=False)

    def scrub_dictionary(self, in_dict: dict, remove_keys: list):
        scrubbed_dict = deepcopy(in_dict)
        for key in remove_keys:
            scrub_key(scrubbed_dict, key)
        if 'round' in remove_keys:
            scrubbed_dict['round'] = self._round
        return scrubbed_dict

    def save_results(self, result_dict: dict):

        self.log_wandb_and_tb(result_dict)
        if self.cfg.save_csv:
            self.save_as_csv(result_dict=result_dict)



    def finalize(self):
        """Save results.
        """
        self.logger.debug(f'[RESULTS] [Round: {self._round:03}] Save results and the global model checkpoint!')
        # df = pd.json_normalize(self.full_results)
        # df.to_csv('results.csv')
        self.writer.close()

        self.logger.info(f' [Round: {self._round:03}] ...finished federated learning!')
        return self.last_result



# @dataclass
# class AllResults:
#     # This might need a default factory
#     round: int = 0
#     participants: dict[str] = field(default_factory=dict)
#     clients_train: ClientResult = field(default_factory=ClientResult)
#     clients_eval: ClientResult = field(default_factory=ClientResult)
#     clients_eval_pre: ClientResult = field(default_factory=ClientResult)
#     # client
#     server_eval: Result = field(default_factory=Result)

    # TBD
    # def calculate_generalization_gap(self):
    #     gen_gap = dict()
    #     curr_res: dict = self.result_dicts[self.round]
    #     for key in curr_res['clients_evaluated_out'].keys():
    #         for name in curr_res['clients_evaluated_out'][key].keys():
    #             if name in ['equal', 'weighted']:
    #                 gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
    #                 gen_gap[f'gen_gap_{key}'] = {name: gap}
    #                 self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
    #                 self.writer.flush()
    #     else:
    #         self.result_dicts[self.round]['generalization_gap'] = dict(gen_gap)



# class Updateable(object):
#     def update(self, new):
#         for key, value in new.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)

