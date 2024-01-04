from src.config import Config
from src.results.resultmanager import Stats
from hydra.utils import to_absolute_path
from dataclasses import asdict, field, dataclass
import pandas as pd
import os
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import platform
import json
import wandb

@dataclass
class FinalResults:
    timestamp: str
    mode: str = field(default='')
    out_dir: str = field(default='')
    hostname: str = field(default=platform.node())
    server_id: str = field(default='')
    client_id: str = field(default='')
    dataset: str = field(default='')
    split: str = field(default='')
    n_clients: int = field(default=-1)
    total_rounds: int = field(default=-1)
    steps: int = field(default=-1)
    opt_cfg: list = field(default_factory=list)
    clients: dict[str, Stats] = field(default_factory=dict)
    server: dict[str, float] = field(default_factory=dict)
    correlation: float = field(default=None)
    runtime: float = field(default=0.0)
    runtime_str: str = field(default='')
    train_sizes: dict[str, int] = field(default_factory=dict)
    eval_sizes: dict[str, int] = field(default_factory=dict)
    server_size: int = field(default=-1)

@dataclass
class FinalResultsCentralized:
    timestamp: str
    mode: str = field(default='')
    out_dir: str = field(default='')
    hostname: str = field(default=platform.node())
    client_id: str = field(default='')
    dataset: str = field(default='')
    split: str = field(default='')
    n_clients: int = field(default=-1)
    total_rounds: int = field(default=-1)
    steps: int = field(default=-1)
    opt_cfg: list = field(default_factory=list)
    correlation: float = field(default=None)
    runtime: float = field(default=0.0)
    runtime_str: str = field(default='')
    train_sizes: dict[str, int] = field(default_factory=dict)
    eval_sizes: dict[str, int] = field(default_factory=dict)

def _finditem(obj:dict, key):
    if key in obj: return key, obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            superkey, item = _finditem(v, key)
            return f'{superkey}.{k}', item
    
def compute_correlatation(arr_1, arr_2):
    assert len(arr_1) ==len(arr_2), "Mismatching array sizes for correlation"
    return pearsonr(arr_1, arr_2)[0]

def post_process(cfg: Config, result: dict, total_time=0.0):

    # Maybe just an omegaconf object would be fine
    # FIXME: Migrate to using new format of dictionary
    # FIXME: Generalize for all forms of simulation

    if cfg.simulator.mode =='federated':
        final = FinalResults(timestamp = datetime.now().strftime("%d/%m/%y, %H:%M:%S"))
        final.out_dir = os.getcwd()
        final.runtime = total_time
        final.runtime_str = str(timedelta(seconds=total_time))
        final.server = result['central_eval']['metrics']
        final.clients = result['local_eval/post_agg']['stats']
        final.n_clients = cfg.simulator.num_clients
        final.dataset = cfg.dataset.name
        final.split = cfg.dataset.split_conf.split_type
        final.total_rounds = cfg.simulator.num_rounds
        final.steps = cfg.simulator.num_rounds * cfg.client.cfg.epochs
        final.mode = cfg.mode
        # FIXME: Find a cleaner way to avoid key errors
        final.eval_sizes = result['local_eval/post_agg']['sizes']
        final.train_sizes = result['local_train/pre_agg']['sizes']
        final.server_size = result['central_eval']['size']

        final.server_id = cfg.server._target_.split('.')[-1].removesuffix('Server').lower()
        final.client_id = cfg.client._target_.split('.')[-1].removesuffix('Client').lower()
    elif cfg.simulator.mode == 'centralized':
        final = FinalResultsCentralized(timestamp = datetime.now().strftime("%d/%m/%y, %H:%M:%S"))
        final.out_dir = os.getcwd()
        final.runtime = total_time
        final.runtime_str = str(timedelta(seconds=total_time))
        final.dataset = cfg.dataset.name
        final.split = cfg.dataset.split_conf.split_type
        final.total_rounds = cfg.simulator.num_rounds
        final.steps = cfg.simulator.num_rounds * cfg.client.cfg.epochs
        final.mode = cfg.mode
        # FIXME:
        # final.eval_sizes = result
        # final.train_sizes = result
        


    if cfg.log_conf:
        flat_cfg = pd.json_normalize(asdict(cfg))
        final.opt_cfg = [f'{key}:{flat_cfg.get(key).values[0]}' for key in cfg.log_conf]        

    result_dictionary = asdict(final)

    if cfg.simulator.use_wandb:
        for key, val in result_dictionary.items():
            wandb.run.summary[key] = val

    with open('final_result.json', 'w') as f:
        json.dump(result_dictionary, f, indent=4)
    df = pd.json_normalize(result_dictionary)

    if not cfg.mode == 'debug':
        filename = to_absolute_path('outputs/consolidated_results.csv')
        if os.path.exists(filename):
            df1 = pd.read_csv(filename)
            df3 = pd.concat([df1,df])
            df3.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)