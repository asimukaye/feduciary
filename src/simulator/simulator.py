

import logging

from src.utils.utils import TqdmToLogger, set_seed
from src.simulator.loaders import load_dataset, load_model
from main import Config
from importlib import import_module
logger = logging.getLogger('[SIMULATOR]')
import torch

# TODO: develop this into an eventual simulator class
'''
 Features this class needs:
 client creation
 server creation
 dataset distribution
 client availability
 client 
'''

class Simulator:
    """Simulator orchestrating the whole process of federated learning.
    """
    def __init__(self, cfg:Config):
        self.round = 0
        self.cfg = cfg
        self.num_clients = cfg.simulator.num_clients
        self.algo = cfg.server.algorithm.name

        self.server_dataset, self.client_datasets = load_dataset(cfg.dataset)

        set_seed(cfg.simulator.seed)
        self.model, model_args = load_model(cfg.model)

    def set_device(self):
        # adjust device
        if 'cuda' in self.cfg.simulator.device:
            assert torch.cuda.is_available(), 'Please check if your GPU is available now!' 
            self.cfg.simulator.device = 'cuda' if self.cfg.simulator.device_ids == [] else f'cuda:{self.cfg.simulator.device_ids[0]}'

    def run_server(self):
        SERVER_CLASS = import_module(f'src.server.{self.algo}', f'{self.algo}')

        server = SERVER_CLASS()
        
        for curr_round in range(1, self.cfg.server.rounds + 1):
            ## update round indicator
            server.round = curr_round

            ## update after sampling clients randomly
            selected_ids = server.update()

            ## evaluate on clients not sampled (for measuring generalization performance)
            if curr_round % self.cfg.simulator.eval_every == 0:
                server.evaluate(excluded_ids=selected_ids)
        else:
            ## wrap-up
            server.finalize()



    def _create_clients(self, client_datasets):
        # Acess the client class
        CLIENT_CLASS = import_module(f'')

        def __create_client(identifier, datasets):
            client = CLIENT_CLASS(self.cfg.client, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[SIM] [Round: {self.round:03}] Creating clients!')
    
        clients = []
        # think of a better way to id the clients
        for id, datasets in TqdmToLogger(enumerate(client_datasets), logger=logger, desc=f'[SIM] [Round: {self.round:03}] ...creating clients...'):
            clients.append(__create_client(id, datasets))            
        
        logger.info(f'[SIM] [Round: {self.round:03}]] ...sucessfully created {self.num_clients} clients!')
        return clients

    
    def finalize(self):
        raise NotImplementedError
