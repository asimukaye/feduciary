
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
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from src.server.baseserver import BaseServer
from src.client.baseclient import BaseClient, model_eval_helper
from src.data import load_vision_dataset
from src.config import Config, SimConfig
from src.utils  import log_tqdm, log_instance
from src.postprocess import post_process
from src.models.model import init_model
from src.results.resultmanager import ResultManager
from src.metrics.metricmanager import MetricManager, Result
from functools import partial
import wandb
logger = logging.getLogger('SIMULATOR')


class Simulator:
    """Simulator orchestrating the whole process of federated learning.
    """
    # TODO: Split into federated, standalone, and centralized simulator
    def __init__(self, cfg:Config):
        self.start_time = time.time()
        self.round: int = 0
        self.master_cfg: Config = cfg
        self.cfg: SimConfig = cfg.simulator
        
        if self.cfg.use_wandb:
            wandb.init(project='fed_ml', job_type=cfg.mode, config=asdict(cfg), resume=True, notes=cfg.desc)

        logger.info(f'[NUM ROUNDS] : {self.cfg.num_rounds}')

        self.num_clients = cfg.simulator.num_clients

        self.set_seed(cfg.simulator.seed)
        # self.algo = cfg.server.algorithm.name

       # # NOTE: THe model spec is being modified in place here.
        self.server_dataset, self.client_datasets = load_vision_dataset(cfg.dataset, cfg.model.model_spec)
        self.model_instance: Module = instantiate(cfg.model.model_spec)

        self.metric_manager = MetricManager(self.master_cfg.client.cfg.eval_metrics, self.round, actor='simulator')
        init_model(cfg.model, self.model_instance)
        self.result_manager = ResultManager(cfg.simulator, logger=logger)
        self.server = None

        self.set_fn_overloads_for_mode()

        self.init_sim()

        self.make_checkpoint_dirs()
        self.is_resumed =False
        
        server_ckpt, client_ckpts = self.find_checkpoint()

        if self.is_resumed:
            logger.info('------------ Resuming training ------------')
            self.load_state(server_ckpt, client_ckpts)
        
        logger.debug(f'Init time: {time.time() - self.start_time} seconds')

    def set_fn_overloads_for_mode(self):
        match self.cfg.mode:
            case 'federated':
                self.init_sim = self.init_federated_mode
                self.run_simulation = self.run_federated_simulation
            case 'standalone':
                self.init_sim = self.init_standalone_mode
                self.run_simulation = self.run_standalone_simulation            
            case 'centralized':
                self.init_sim = self.init_centralized_mode
                self.run_simulation = self.run_centralized_simulation
            case _:
                raise AssertionError(f'Mode: {self.cfg.mode} is not implemented')

    def init_federated_mode(self):
        # model_instance: Module = instantiate(self.master_cfg.model.model_spec)

        server_partial: partial = instantiate(self.master_cfg.server)
        self.clients: dict[str, BaseClient] = defaultdict(BaseClient)

        
        # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
        self.clients = self._create_clients(self.client_datasets, copy.deepcopy(self.model_instance))

        self.all_client_ids = list(self.clients.keys())
        # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary
        self.server: BaseServer = server_partial(model=self.model_instance, dataset=self.server_dataset, clients= self.clients, result_manager=self.result_manager)

    def init_standalone_mode(self):
        self.clients: dict[str, BaseClient] = defaultdict(BaseClient)

        # NOTE: THe model spec is being modified in place here.
        # self.central_dataset, client_datasets= load_vision_dataset(self.master_cfg.dataset, self.master_cfg.model.model_spec)


        # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
        self.clients = self._create_clients(self.client_datasets, copy.deepcopy(self.model_instance))

    def init_centralized_mode(self):
        pass

    def make_checkpoint_dirs(self):
        os.makedirs('server_ckpts', exist_ok = True)
        for cid in self.clients.keys():
            os.makedirs(f'client_ckpts/{cid}', exist_ok = True)

    def find_checkpoint(self)-> tuple[list, list]:
        server_ckpts = sorted(glob.glob('server_ckpts/server_ckpt_*'))
        client_ckpts = {}

        for dir in os.listdir('client_ckpts/'):
            files = sorted(os.listdir(f'client_ckpts/{dir}'))
            if files:
                client_ckpts[dir] = f'client_ckpts/{dir}/{files[-1]}'
                logger.info(f'------ Found client {dir} checkpoint: {client_ckpts[dir]} ------')
        if server_ckpts or client_ckpts:
            self.is_resumed = True
            if server_ckpts:
                logger.info(f'------ Found server checkpoint: {server_ckpts[-1]} ------')
                return server_ckpts[-1], client_ckpts
            else:
                return [], client_ckpts
        else:
            logger.debug('------------ No checkpoints found. Starting afresh ------------')
            self.is_resumed = False
            return [], []

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
    
    def central_evaluate_clients(self, cids: list[str]):
        '''Evaluate the clients on servers holdout set'''
        # NOTE: this is useless if the central evaluation is done after the last aggregation
        test_loader = DataLoader(dataset=self.server_dataset, batch_size=self.master_cfg.client.cfg.batch_size)
        eval_result = {}
        # FIXME: Might need to instantiate globally elsewhere
        client_cfg_inst = instantiate(self.master_cfg.client.cfg)
        for cid in cids:
            eval_result[cid] = model_eval_helper(self.clients[cid].model, test_loader, client_cfg_inst, self.metric_manager, self.round)
        
        self.result_manager.log_client_result(eval_result, 'central_eval')
        
    def local_evaluate_clients(self, cids: list[str]):
        # Local evaluate the clients on their test sets
        eval_results = self.server._eval_request(cids)

        self.result_manager.log_client_result(eval_results, event='client_eval_post')

    def run_standalone_simulation(self):
        for curr_round in range(self.round, self.cfg.num_rounds + 1):
            train_result = {}
            eval_result = {}

            for cid, client in self.clients.items():
                logger.info(f'-------- Client: {cid} --------\n')
                client.round = curr_round
                train_result[cid] = client.train()
                eval_result[cid] = client.evaluate()
            if curr_round % self.cfg.checkpoint_every == 0:
                self.save_checkpoints()

            self.result_manager.log_client_result(train_result, 'client_train')
            self.result_manager.log_client_result(eval_result, 'client_eval')

            self.result_manager.update_round_and_flush(curr_round)

    def run_centralized_simulation(self):
        # TBD
        pass

    def run_federated_simulation(self):
        '''Runs the simulation in federated mode'''
        # self.clients = self._create_clients(self.client_datasets)
        # self.server.initialize(clients, )

        for curr_round in range(self.round, self.cfg.num_rounds +1):
            logger.info(f'-------- Round: {curr_round} --------\n')
            # wandb.log({'round': curr_round})
            loop_start = time.time()
            self.round = curr_round
            ## update round indicator
            self.server.round = curr_round

            ## update after sampling clients randomly
            update_ids = self.server.update()

            ## evaluate on clients not sampled (for measuring generalization performance)
            if curr_round % self.master_cfg.server.cfg.eval_every == 0:
                # Can have specific evaluations later
                eval_ids = self.all_client_ids
                self.server._broadcast_models(eval_ids)
                self.local_evaluate_clients(eval_ids)
                # self.central_evaluate_clients(eval_ids)
                # self.server.reset_client_models(eval_ids)


                self.server.server_evaluate()

            if curr_round % self.cfg.checkpoint_every == 0:
                self.save_checkpoints()
            
            # This is weird, needs some rearch
            self.result_manager.update_round_and_flush(curr_round)

            loop_end = time.time() - loop_start
            logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')


    def save_checkpoints(self):
        if self.server:
            self.server.save_checkpoint()
        if self.clients:
            for client in self.clients.values():
                client.save_checkpoint()

    # TODO: Test method to load client checkpoints
    def load_state(self, server_ckpt_path: str, client_ckpts: dict):
        if server_ckpt_path:
            self.server.load_checkpoint(server_ckpt_path)
            self.round = self.server.round
        if client_ckpts:
            for cid, ckpt in client_ckpts.items():
                self.clients[cid].load_checkpoint(ckpt)
            if self.round == 0:
                self.round = self.clients[cid].round


    @log_instance(attrs=['round', 'num_clients'], m_logger=logger)
    def _create_clients(self, client_datasets, model: Module) -> dict[str, BaseClient]:
        # Acess the client clas
        client_partial = instantiate(self.master_cfg.client)

        def __create_client(idx, datasets, model):
            client: BaseClient = client_partial(id_seed=idx, dataset=datasets, model=model)
            return client.id, client

        clients = {}
        # think of a better way to id the clients
        for idx, datasets in log_tqdm(enumerate(client_datasets), logger=logger, desc=f'[Round: {self.round:03}] creating clients '):
            client_id, client_obj = __create_client(idx, datasets, model)
            clients[client_id] = client_obj
        return clients

       
    def finalize(self):
        if self.server:
            self.server.finalize()
        final_result = self.result_manager.finalize()
        total_time= time.time() - self.start_time
        logger.info(f'Total runtime: {total_time} seconds')
        post_process(self.master_cfg, final_result, total_time=total_time)

        del self.clients
        del self.server
        logger.info('Closing Feduciary Simulator')
