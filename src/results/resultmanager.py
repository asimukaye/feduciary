
from collections import defaultdict
from torch import Tensor
from torch.nn import Parameter
import numpy as np
import typing as t
from torch.utils.tensorboard import SummaryWriter
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from logging import Logger
import pandas as pd
import wandb
from copy import deepcopy
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
    # TODO: Find a way to incorporate below
    # lump_dim_name: str = ''  # An optional indicator string for what is being lumped in this stat calculation

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


# 
class ResultManager:
    '''Accumulate and process the results.
    Some general notations:
    Metric: What is to be logged/plotted. Eg: accuracy, loss, parameters...
    Event: What is the context: Eg: local_train, local_eval, central_eval...
    Phase: When was this logged in the master loop. Eg: pre_aggregation...
    Actor: Who supplied this metric. Eg: Server, Client_0, ...; can also be used to give stats like client_mean, client_max, etc.
    Metric, phase, and actor are mandatory keys for a metric, whereas event can be optionally used to disambiguate an ambiguous case'''

    _round: int = 0
    # TODO: Migrate to metric event actor dictionaries everywhere
    # TODO: Evolve into a singleton class for evaluation purposes.

    def __init__(self, cfg: SimConfig, logger: Logger) -> None:
        # Event actor metric form dictionaries
        self.result_dict = defaultdict()
        self.last_result = defaultdict()

        # Metric Event Actor dictionary with the phase lumped along with the event
        self.metric_event_actor_dict: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))

        # Metric Event Actor Phase dictionary with the phase distinctly listed out at the end
        self.metric_event_actor_phase_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # self.metric_event_actor_phase_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        self.phase_tracker: dict(int) = dict()
        self._phase_counter = 0
        self._step_counter = 0
        
        # TODO: Remove persistent dict if unnecessary. long running dictionary of metrics, event, actor for plotting
        self.persistent_dict = defaultdict(lambda: defaultdict(list))

        self.writer = SummaryWriter()
        self.logger = logger
        self.cfg = cfg

    def get_client_results(self,result:dict[str, Result], event: str, phase: str) -> ClientResult:
        client_result = ClientResult()
        client_result.results = result
        client_result.round = self._round
        client_result = self.compute_client_stats_and_log(client_result, event=event, phase=phase)
        return client_result

    @staticmethod    
    def compute_stats(array: list, lump_dim_name: str = '') -> Stats:
        stat = Stats(-1, -1, -1, -1)
        np_array = np.array(array).astype(float)
        stat.mean = np.mean(np_array)
        stat.std = np.std(np_array)
        stat.minimum = np.min(np_array)
        stat.maximum = np.max(np_array)
        # stat.lump_dim_name = lump_dim_name
        return stat

    def _round_check(self, rnd, cid):
        assert self._round == rnd, f'Mismatching rounds: {cid} :r {rnd}, global round: {self._round}'

    def compute_client_stats_and_log(self, client_result: ClientResult,  event: str, phase: str) -> ClientResult:

        dict_by_metric = defaultdict(dict)
        # Maybe per client logging could be disabled later
        for cid, clnt_result in client_result.results.items():
            self._round_check(clnt_result.round, cid)

            for metric, value in clnt_result.metrics.items():
                # metrics_list[metric].append(value)
                dict_by_metric[metric][cid] = value

            dict_by_metric['size'][cid] = clnt_result.size
            client_result.sizes[cid] = clnt_result.size

            log_metric(f'client:{cid}_{event}', self._round, clnt_result.metrics, self.logger)

        # for metric, vals in metrics_list.items():
        for metric, vals in dict_by_metric.items():

            # client_result.stats[metric] = self.compute_stats(vals)
            client_result.stats[metric] = self.compute_stats(list(vals.values()))

            log_metric(f'{event}_{metric}', self._round, client_result.stats[metric].__dict__, self.logger)
        
        for metric, stat_obj in client_result.stats.items():
            for stat_name, value in asdict(stat_obj).items():
                dict_by_metric[metric][stat_name] = value

        for metric in dict_by_metric.keys():
            for actor, value in dict_by_metric[metric].items():
                self._add_metric(metric, event, phase, actor, value)
            # self.metric_event_actor_dict[metric][event] = dict_by_metric[metric]
        return client_result
    
    def log_general_result(self, result: Result, phase: str, actor:str, event: str):
        
        self._round_check(result.round, actor)

        # TODO: Make the below two lines obsolete
        log_metric(event, result.round, result.metrics, self.logger)
        self.result_dict[event] = asdict(result)

        # event = 'central_eval'

        for metric, value in result.metrics.items():
            self._add_metric(metric, event, phase, actor, value)
            # self.metric_event_actor_dict[metric][event]['server'] = value
        
        self._add_metric('size', event, phase,  actor, result.size)
        
        # self.metric_event_actor_dict['size'][event]['server'] =  result.size
        

    def log_clients_result(self, result: dict[Result],  phase: str, event='local_eval'):

        client_result = self.get_client_results(result, event=event, phase=phase)

        self.result_dict[f'{event}/{phase}'] = asdict(client_result)
        # self.result.participants[key] = list(result.keys())
        return client_result


    def log_duplicate_parameters_for_clients(self, cids: t.Union[str, t.List], phase: str ,event: str = '', reference_actor='server'):
        # Helper function to duplicate the parameters to avoid expensive operations
        if isinstance(cids, str):
            cids = [cids]

        for metric in self.metric_event_actor_dict.keys():
            if 'param'in metric.split('/'):
                reference_value = self.metric_event_actor_dict[metric][f'{event}/{phase}'][reference_actor]
    
                for cid in cids:
                    self._add_metric(metric, event, phase, cid, reference_value)


    def log_parameters(self, model_params: dict[str, Parameter], phase: str, actor: str, event: str = '' , metric= 'param', verbose=False) -> dict:
        # LUMP The parameters layer wise and add to the result dictionary
        out_dict = {}
        avg = 0.0
        weighted_avg = 0.0
        layer_dims = 0
        # model_params = deepcopy(model_params_in)
        for name, param in model_params.items():
            param_detached = param.detach().cpu()
            layer_wise_abs_mean = param_detached.abs().mean().item()
            avg +=layer_wise_abs_mean
            layer_dim = np.prod(param_detached.size())
            weighted_avg += layer_wise_abs_mean*layer_dim
            layer_dims += layer_dim
            if verbose:
                out_dict[name]= layer_wise_abs_mean

        avg = avg/len(model_params)
        weighted_avg = weighted_avg/layer_dims

        out_dict['avg'] = avg
        out_dict['wtd_avg'] = weighted_avg

        # self.result_dict[event] = out_dict
        for k, val in out_dict.items():
            self._add_metric(f'{metric}/{k}', event, phase, actor, val)
        return out_dict

    
    def _add_metric(self, metric, event, phase, actor, value):
        # Metric addition of what metric, what contxt, when and who and what value

        self.metric_event_actor_phase_dict[metric][f'{event}/concat'][actor][phase] = value
        # This code assigns phase ordering based on when it was called from the call stack. This may or may not be the same as algorithm order
        if phase not in self.phase_tracker:
            self.phase_tracker[phase] = self._phase_counter
            self._phase_counter += 1
        self.metric_event_actor_dict[metric][f'{event}/{phase}'][actor] = value


    def log_general_metric_average(self, metrics: dict, metric_name: str, actor: str, phase: str, event: str = 'avg', weights=None):
        # Helper function to compute a weighted average over dictionary values
        assert isinstance(metrics, dict)
        if weights is None:
            weights = [1.0/len(metrics) for k in metrics.keys()]
        
        assert len(weights) == len(metrics)
        average = 0.0
        for w, v in zip(weights, metrics.values()):
            average += w*v
        
        self._add_metric(metric_name, event, phase, actor, average)


    def log_general_metric(self, metric_val, metric_name: str, actor: str, phase: str, event: str = ''):
        
        if isinstance(metric_val, dict):
            for key, val in metric_val.items():
                self.log_general_metric(val, f'{metric_name}/{key}', actor, phase, event)
        elif isinstance(metric_val, (float, int)):
            self._add_metric(metric_name, event, phase, actor, metric_val)
        else:
            err_str = f'Metric logging for {metric_name} of type: {type(metric_val)} is not supported'
            logger.error(err_str)
            raise TypeError(err_str)
    
    def update_round_and_flush(self, rnd:int):
        self.result_dict['round'] = self._round
        self.metric_event_actor_dict['round'] = self._round

        self.last_result = deepcopy(self.result_dict)
        self.save_results(self.result_dict)
        self.result_dict = defaultdict()
        self.metric_event_actor_dict = defaultdict(lambda: defaultdict(dict))

        # Metric Event Actor Phase dictionary with the phase distinctly listed out at the end
        self.metric_event_actor_phase_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        self._round = rnd+1  # setup for the next round


    def log_wandb_and_tb(self, result_dict: dict, tag='results'):
        # TODO: Make this function cleaner

        remove_keys = ['actor', 'epoch', 'sizes',
                        'size', 'round']
        # scrubbed_dict = self.scrub_dictionary(result_dict, remove_keys)

        scrubbed_dict = self.scrub_dictionary(self.metric_event_actor_dict, remove_keys)

        # HACK: Experimental code to add multi step logging
        scrubbed_dict_2 = self.scrub_dictionary(self.metric_event_actor_phase_dict, remove_keys, add_round_back=False)

        flattened_dict_2 = pd.json_normalize(scrubbed_dict_2, sep='/', max_level=2).to_dict('records')[0]
        # ic(flattened_dict_2)
        # ic(len(flattened_dict_2))

        # ic(self.phase_tracker)
        # ic(self._phase_counter)

        # sorted_phases = dict(sorted(self.phase_tracker.items(), key=lambda x:x[1]))

        phase_wise_sorted = [{} for i in range(self._phase_counter)]
        for key, phase_value_dict in flattened_dict_2.items():
            assert isinstance(phase_value_dict, dict)
            if len(phase_value_dict) >1:
                for phase, val in phase_value_dict.items():
                    phase_wise_sorted[self.phase_tracker[phase]][key] = val
                    
        if self.cfg.use_wandb:
            for i, dicts in enumerate(phase_wise_sorted):
                wandb.log(dicts, step=self._step_counter + i, commit=False)
        

        self._step_counter += len(phase_wise_sorted) -1
        # Important step to keep track of the steps accurately for final logging
        # self._step_counter += self._phase_counter

        
        # Fully flatten the results

        flattened_dict = pd.json_normalize(scrubbed_dict, sep='/').to_dict('records')[0]

        if self.cfg.use_tensorboard:
            self.writer.add_scalars(tag, flattened_dict, global_step=self._round)
            self.writer.flush()
        
        # Add figures on top

        if self.cfg.use_wandb:
            # wandb_dict = self.wandb_plots()

            wandb.log(flattened_dict, step=self._step_counter, commit=True)
            # Making sure the step counter of wandb matches the locally tracked one. Note: Committing increases the step value
            self._step_counter += 1

            # wandb.log(wandb_dict)


    def save_as_csv(self, result_dict: dict, filename='results.csv'):
        df = pd.json_normalize(result_dict)
        if self._round == 0:
            df.to_csv(filename, mode='w', index=False, header=True)
        else:
            df.to_csv(filename, mode='a', index=False, header=False)

    def scrub_dictionary(self, in_dict: dict, remove_keys: list, add_round_back = True):
        scrubbed_dict = deepcopy(in_dict)
        for key in remove_keys:
            scrub_key(scrubbed_dict, key)
        if 'round' in remove_keys:
            if add_round_back:
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


    # Some valiant effort functions, maybe useful later
    def fuse_events(self, metric: str, past_event: str, present_event: str, actor: str, x, x_key:str ,delta_x=0.5, actor_2: str = None, y_key: str =''):
        # TODO: Think of a better way to do this function

        if actor_2 is None:
            actor_2 = actor
        y_past = self.metric_event_actor_dict[metric][past_event][actor]
        y_present = self.metric_event_actor_dict[metric][present_event][actor_2]
        self.persistent_dict[f'{metric}/{actor}'][x_key].extend([x, x + delta_x])
        self.persistent_dict[f'{metric}/{actor}'][metric].extend([y_past, y_present])

    def add_matplotlib_plot(self, input_dict: dict, title: str = ''):
        fig = plt.figure()
        ax = fig.add_subplot()
        iterate = iter(input_dict.items())
        x_label, x_vals = next(iterate)
        y_label, y_vals = next(iterate)
        ax.clear()
        ax.plot(x_vals, y_vals, label=y_label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        return fig
  
    def wandb_multi_line_plots(self, plot_keys: list =None):
        # What a waste of time this was...
        wandb_dict = {}
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


# @dataclass
# class AllResults:
#     # This might need a default factory
#     round: int = 0
#     participants: dict[str] = field(default_factory=dict)
#     clients_train: ClientResult = field(default_factory=ClientResult)
#     clients_eval: ClientResult = field(default_factory=ClientResult)
#     clients_eval_pre: ClientResult = field(default_factory=ClientResult)
#     # client
#     central_eval: Result = field(default_factory=Result)

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

