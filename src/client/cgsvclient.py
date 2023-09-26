import copy

from .fedavgclient import FedavgClient
from src import MetricManager


class CgsvClient(FedavgClient):
    def __init__(self, **kwargs):
        super(CgsvClient, self).__init__(**kwargs)
    
    def upload(self):
        # Upload the model back to the server
        self.model.to('cpu')
        return self.model.parameters()
    # TODO: add this later/
    # def download(self, addendum):
    #     # As opposed to FedAvg, the client receives gradients rather than model parameters
    #     self.model.gradi