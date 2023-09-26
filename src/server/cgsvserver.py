import logging
import torch
from torch.optim.lr_scheduler import ExponentialLR
# from .fedavgserver import FedavgServer
from .baseserver import BaseServer
from ..algorithm.cgsv import CgsvOptimizer

logger = logging.getLogger(__name__)

class CgsvServer(BaseServer):
    def __init__(self, **kwargs):
        super(CgsvServer, self).__init__(**kwargs)
        
        # self.server_optimizer = self._get_algorithm(self.model, lr=self.args.lr, gamma=self.args.gamma)

        self.importance_coefficients = dict.fromkeys(self._clients, 0.0)

        self.server_optimizer = CgsvOptimizer(params=self.model.parameters(), client_ids=self._clients.keys(), lr=self.args.lr, gamma=self.args.gamma, alpha=self.args.alpha)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.server_optimizer, gamma=self.args.lr_decay)

        # self.importance_coefficients = dict.fromkeys(self._clients, 0.0)
        # self.alpha = self.args.alpha
        # self._init_coefficients()

 
    # def _update_coefficients(self, id, cgsv):
    #     self.importance_coefficients[id] = self.alpha * self.importance_coefficients[id] + (1 - self.alpha)* cgsv


    def _aggregate(self, ids, updated_sizes):
        # Calls client upload and server accumulate
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # calculate importance coefficients according to sample sizes
        # coefficients = {identifier: float(coefficient / sum(updated_sizes.values())) for identifier, coefficient in updated_sizes.items()}

        # coefficients = self._update_coefficients(ids, cgsv)
        
        # accumulate weights
        for identifier in ids:
            local_weights_itr = self._clients[identifier].upload()
            
            # Compute Gradient
            # cgsv = self.server_optimizer._compute_cgsv(identifier)
            # self._update_coefficients(identifier, cgsv)
            # Accumulate weights
            self.server_optimizer.accumulate(local_weights_itr, identifier)

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    def update(self):
        """Update the global model through federated learning.
        """
        # randomly select clients
        selected_ids = self._sample_clients()

        #TODO: Sparsify gradients here
        # self._sparsify_gradients(selected_ids)

        # broadcast the current model at the server to selected clients
        self._broadcast_models(selected_ids)
        
        # request update to selected clients
        updated_sizes = self._request(selected_ids, eval=False)
        
        # print("Update sizes type: ", type(updated_sizes))
        # print("Update sizes: ", updated_sizes)

        # request evaluation to selected clients
        self._request(selected_ids, eval=True, participated=True)

        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad() # empty out buffer

        self._aggregate(selected_ids, updated_sizes) # aggregate local updates
        
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate

        # remove model copy in clients
        self._cleanup(selected_ids)

        return selected_ids
