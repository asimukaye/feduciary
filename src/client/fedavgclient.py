import copy
import torch
import inspect

from torch.utils.data import DataLoader
from .baseclient import BaseClient
from src.metrics.metricmanager import MetricManager

class FedavgClient(BaseClient):
    def __init__(self, id_seed, cfg, dataset):
        super(FedavgClient, self).__init__()