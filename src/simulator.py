
import os
from importlib import import_module

import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate

from src.server.baseserver import BaseServer
from src.client.baseclient import BaseClient
from src.datasets.utils.data import load_vision_dataset
from src.config import Config, SimConfig
from src.utils  import TqdmToLogger, log_instance


# TODO: develop this into an eventual simulator class


logger = logging.getLogger('SIMULATOR')



class Simulator:
    """Simulator orchestrating the whole process of federated learning.
    """
    def __init__(self, cfg:Config):
        self.round:int = 0
        self.master_cfg:Config = cfg
        self.cfg:SimConfig = cfg.simulator

        self.num_clients = cfg.simulator.num_clients
        # self.algo = cfg.server.algorithm.name

        server_partial = instantiate(cfg.server)
        self.clients: dict(str, BaseClient) = {}
        # print(server_partial)

        server_dataset, client_datasets= load_vision_dataset(cfg.dataset, cfg.model.model_spec)

        # print(cfg.model.model_spec)
        self.set_seed(cfg.simulator.seed)

        # model_instance, model_args = load_model(cfg.model)

        model_instance = instantiate(cfg.model.model_spec)

        self.writer = SummaryWriter()

        self.clients = self._create_clients(client_datasets)

        # print(self.clients)

        # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary
        self.server: BaseServer = server_partial(model=model_instance, dataset=server_dataset, clients= self.clients, writer=self.writer)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f'[SEED] ...seed is set: {seed}!')
    
    def run_simulation(self):
        '''Runs the simulation'''
        # self.clients = self._create_clients(self.client_datasets)
        # self.server.initialize(clients, )

        for curr_round in range(self.cfg.num_rounds + 1):
            ## update round indicator
            self.server.round = curr_round

            ## update after sampling clients randomly
            selected_ids = self.server.update()

            ## evaluate on clients not sampled (for measuring generalization performance)
            if curr_round % self.master_cfg.server.cfg.eval_every == 0:
                self.server.evaluate(excluded_ids=selected_ids)
            
            self.server.result_manager.update_round_and_flush(curr_round)
        self.server.finalize()
  
    @log_instance(attrs=['round', 'num_clients'], m_logger=logger)
    def _create_clients(self, client_datasets):
        # Acess the client clas
        client_partial = instantiate(self.master_cfg.client)

        def __create_client(idx, datasets):
            client:BaseClient = client_partial(id_seed=idx, dataset=datasets)
            return client.id, client

        clients = {}
        # think of a better way to id the clients
        for idx, datasets in TqdmToLogger(enumerate(client_datasets), logger=logger,
        desc=f'[Round: {self.round:03}] ...creating clients...'):
            client_id, client = __create_client(idx, datasets)
            clients[client_id] = client       

        return clients


    # def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        # randomly select all remaining clients not participated in current round
        # selected_ids = self._sample_selected_clients(exclude=excluded_ids)
        # self._broadcast_models(selected_ids)

        # # request evaluation 
        # ## `local`: evaluate on selected clients' holdout set
        # ## `global`: evaluate on the server's global holdout set 
        # ## `both`: conduct both `local` and `global` evaluations
        # if self.args.eval_type == 'local':
        #     self._request(selected_ids, eval=True, participated=False)
        # elif self.args.eval_type == 'global':
        #     self._central_evaluate()
        # elif self.args.eval_type == 'both':
        #     self._request(selected_ids, eval=True, participated=False)
        #     self._central_evaluate()

        # remove model copy in clients
        # self._cleanup(selected_ids)

       
    def finalize(self):
        raise NotImplementedError
