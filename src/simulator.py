
import os
import glob
from collections import defaultdict
from dataclasses import asdict
import copy
import time
import logging
import random
import numpy as np
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate
from src.server.baseserver import BaseServer
from src.client.baseclient import BaseClient
from src.datasets.utils.data import load_vision_dataset
from src.config import Config, SimConfig
from src.utils  import log_tqdm, log_instance
from src.postprocess import post_process
from src.models.model import init_model
from src.results.resultmanager import AllResults, ResultManager
from functools import partial
import wandb
logger = logging.getLogger('SIMULATOR')


class Simulator:
    """Simulator orchestrating the whole process of federated learning.
    """
    def __init__(self, cfg:Config):
        self.start_time = time.time()
        self.round: int = 0
        self.master_cfg: Config = cfg
        self.cfg: SimConfig = cfg.simulator
        # self.tb_server = Process()

        self.make_checkpoint_dirs()

        self.is_resumed =False
        
        ckpt = self.find_checkpoint()
        if self.cfg.use_wandb:
            wandb.init(project='fed_ml', job_type=cfg.mode, config=asdict(cfg), resume=True)

        logger.info(f'[NUM ROUNDS] : {self.cfg.num_rounds}')

        self.num_clients = cfg.simulator.num_clients

        self.set_seed(cfg.simulator.seed)
        # self.algo = cfg.server.algorithm.name

        server_partial: partial = instantiate(cfg.server)
        self.clients: dict[str, BaseClient] = defaultdict(BaseClient)
        # print(server_partial)


        # NOTE: THe model spec is being modified in place here.
        server_dataset, client_datasets= load_vision_dataset(cfg.dataset, cfg.model.model_spec)

        # print(cfg.model.model_spec)
        # model_instance, model_args = load_model(cfg.model)

        model_instance: Module = instantiate(cfg.model.model_spec)

        init_model(cfg.model, model_instance)

        # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
        self.clients = self._create_clients(client_datasets, copy.deepcopy(model_instance))

        # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary

        self.result_manager = ResultManager(cfg.simulator, logger=logger)
        self.server: BaseServer = server_partial(model=model_instance, dataset=server_dataset, clients= self.clients, result_manager=self.result_manager)

        if self.is_resumed:
            logger.info('------------ Resuming training ------------')
            self.load_state(ckpt)

        
        logger.debug(f'Init time: {time.time() - self.start_time} seconds')

    def make_checkpoint_dirs(self):
        os.makedirs('server_ckpts', exist_ok = True)
        os.makedirs('client_ckpts', exist_ok = True)

    def find_checkpoint(self)-> bool:
        ckpts = sorted(glob.glob('server_ckpts/server_ckpt_*'))
        if ckpts:
            logger.info(f'------ Found checkpoint: {ckpts[-1]} ------')
            self.is_resumed = True
            return ckpts[-1]
        else:
            logger.debug('------------ No checkpoints found. Starting afresh ------------')
            self.is_resumed = False
            return None


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

        for curr_round in range(self.round, self.cfg.num_rounds +1):
            logger.info(f'-------- Round: {curr_round} --------\n')
            # wandb.log({'round': curr_round})
            self.round = curr_round
            ## update round indicator
            self.server.round = curr_round

            ## update after sampling clients randomly
            selected_ids = self.server.update()

            ## evaluate on clients not sampled (for measuring generalization performance)
            if curr_round % self.master_cfg.server.cfg.eval_every == 0:
                self.server.evaluate(excluded_ids=selected_ids)

            if curr_round % self.cfg.checkpoint_every == 0:
                self.save_checkpoints()
            
            self.server.result_manager.update_round_and_flush(curr_round)


    def save_checkpoints(self):
        self.server.save_checkpoint()
        # self.clients.save_checkpoint()

    # TODO: Add method to load client checkpoints also
    def load_state(self, server_ckpt_path, client_ckpt_path=None):

        self.server.load_checkpoint(server_ckpt_path)
        self.round = self.server.round


    @log_instance(attrs=['round', 'num_clients'], m_logger=logger)
    def _create_clients(self, client_datasets, model: Module):
        # Acess the client clas
        client_partial = instantiate(self.master_cfg.client)

        def __create_client(idx, datasets, model):
            client:BaseClient = client_partial(id_seed=idx, dataset=datasets, model=model)
            return client.id, client

        clients = {}
        # think of a better way to id the clients
        for idx, datasets in log_tqdm(enumerate(client_datasets), logger=logger, desc=f'[Round: {self.round:03}] creating clients '):
            client_id, client = __create_client(idx, datasets, model)
            clients[client_id] = client       

        return clients

       
    def finalize(self):
        result = self.server.finalize()
        total_time= time.time() - self.start_time
        logger.info(f'Total runtime: {total_time} seconds')
        post_process(self.master_cfg, result, total_time=total_time)

        del self.clients
        del self.server
        logger.info('Closing Feduciary')
