from src.config import Config
from src.metrics.metricmanager import Stats, AllResults
from hydra.utils import to_absolute_path
from dataclasses import asdict, field, dataclass
import pandas as pd
import os
from scipy.stats import pearsonr
from datetime import datetime

@dataclass
class FinalResults:
    timestamp: str = field(default='')
    mode: str = field(default='')
    server_id: str = field(default='')
    client_id: str = field(default='')
    dataset: str = field(default='')
    split: str = field(default='')
    n_clients: int = field(default=-1)
    rounds: int = field(default=-1)
    steps: int = field(default=-1)
    opt_cfg: list = field(default_factory=list)
    clients: dict[str, Stats] = field(default_factory=dict)
    server: dict[str, float] = field(default_factory=dict)
    correlation: float = field(default=float('nan'))


def _finditem(obj:dict, key):
    if key in obj: return key, obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            superkey, item = _finditem(v, key)
            return f'{superkey}.{k}', item
    
def compute_correlatation(arr_1, arr_2):
    assert len(arr_1) ==len(arr_2), "Mismatching array sizes for correlation"
    return pearsonr(arr_1, arr_2)[0]

def post_process(cfg: Config, result:AllResults):

    final = FinalResults()
    final.timestamp = datetime.now()
    final.server = result.server_eval.metrics
    final.clients = result.clients_eval.stats
    final.n_clients = cfg.simulator.num_clients
    final.dataset = cfg.dataset.name
    final.split = cfg.dataset.split_type
    final.rounds = cfg.simulator.num_rounds
    final.steps = cfg.simulator.num_rounds * cfg.client.cfg.epochs
    final.mode = cfg.mode
    final.server_id = cfg.server._target_.split('.')[-1].removesuffix('Server').lower()
    final.client_id = cfg.client._target_.split('.')[-1].removesuffix('Client').lower()

    if cfg.log_conf:
        flat_cfg = pd.json_normalize(asdict(cfg))
        final.opt_cfg = [f'{key}:{flat_cfg.get(key).values[0]}' for key in cfg.log_conf]        

    df = pd.json_normalize(asdict(final))

    filename = to_absolute_path('outputs/consolidated_results.csv')

    if os.path.exists(filename):
        df1 = pd.read_csv(filename)
        df3 = pd.concat([df1,df],axis=1,join='outer')
        df3.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)