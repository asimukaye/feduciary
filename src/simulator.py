
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
from src.split import get_client_datasets

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
            wandb.init(project='fed_ml', job_type=cfg.mode,
                        config=asdict(cfg), resume=True, notes=cfg.desc)

        logger.info(f'[SIM MODE] : {self.cfg.mode}')
        logger.info(f'[SERVER] : {self.master_cfg.server._target_.split(".")[-1]}')
        logger.info(f'[CLIENT] : {self.master_cfg.client._target_.split(".")[-1]}')

        logger.info(f'[NUM ROUNDS] : {self.cfg.num_rounds}')

        self.num_clients = cfg.simulator.num_clients

        self.set_seed(cfg.simulator.seed)
        # self.algo = cfg.server.algorithm.name
        self.init_dataset_and_model()

        self.metric_manager = MetricManager(cfg.client.cfg.eval_metrics, self.round, actor='simulator')
        self.result_manager = ResultManager(cfg.simulator, logger=logger)
        self.server = None

        self.set_fn_overloads_for_mode()

        self.init_sim()

        # TODO: consolidate checkpointing and resuming logic systematically

        self.make_checkpoint_dirs()
        self.is_resumed =False
        
        server_ckpt, client_ckpts = self.find_checkpoint()

        if self.is_resumed:
            logger.info('------------ Resuming training ------------')
            self.load_state(server_ckpt, client_ckpts)
        
        logger.debug(f'Init time: {time.time() - self.start_time} seconds')

    def init_dataset_and_model(self):
        '''Initialize the dataset and the model here'''
        # NOTE: THe model spec is being modified in place here.
        # TODO: Generalize this logic for all datasets
        self.test_set, self.train_set, dataset_model_spec  = load_vision_dataset(self.master_cfg.dataset)

        model_spec = self.master_cfg.model.model_spec
        if model_spec.in_channels is None:
            model_spec.in_channels = dataset_model_spec.in_channels
            logger.info(f'[MODEL CONFIG] Setting model in channels to {model_spec.in_channels}')
        else:
            logger.info(f'[MODEL CONFIG] Overriding model in channels to {model_spec.in_channels}')

        
        if model_spec.num_classes is None:
            model_spec.num_classes = dataset_model_spec.num_classes
            logger.info(f'[MODEL CONFIG] Setting model num classes to {model_spec.num_classes}')
        else:
            logger.info(f'[MODEL CONFIG] Overriding model num classes to {model_spec.num_classes}')


        self.master_cfg.model.model_spec = model_spec


        # self.model_instance: Module = instantiate(cfg.model.model_spec)
        self.model_instance = init_model(self.master_cfg.model)


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
        
        #  Server gets the test set
        self.server_dataset = self.test_set
        # Clients get the splits of the train set with an inbuilt test set
        self.client_datasets =  get_client_datasets(self.master_cfg.dataset.split_conf, self.train_set)

        
        # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
        self.clients = self._create_clients(self.client_datasets, copy.deepcopy(self.model_instance))

        # HACK: Fixing batch size for imbalanced client case
        # FIXME: Formalize this later
        if self.master_cfg.dataset.split_conf.split_type == 'one_imbalanced_client':
            self.clients['0000'].cfg.batch_size = int(self.clients['0000'].cfg.batch_size/2)
            for cid, cl in self.clients.items():
                logger.debug(f'[BATCH SIZES:] CID: {cid}, batch size: {cl.cfg.batch_size}')

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
        trainer: BaseClient = instantiate(self.master_cfg.client)

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
        
        logger.info(f'[SEED] Simulator global seed is set to: {seed}!')
    
    def central_evaluate_clients(self, cids: list[str]):
        '''Evaluate the clients on servers holdout set'''
        # NOTE: this is useless if the central evaluation is done after the last aggregation
        test_loader = DataLoader(dataset=self.server_dataset, batch_size=self.master_cfg.client.cfg.batch_size)
        eval_result = {}
        # FIXME: Might need to instantiate globally elsewhere
        client_cfg_inst = instantiate(self.master_cfg.client.cfg)
        for cid in cids:
            eval_result[cid] = model_eval_helper(self.clients[cid].model, test_loader, client_cfg_inst, self.metric_manager, self.round)
        
        self.result_manager.log_clients_result(eval_result, phase='post_agg', event='central_eval')
        
    def local_evaluate_clients(self, cids: list[str]):
        # Local evaluate the clients on their test sets
        eval_results = self.server._eval_request(cids)

        self.result_manager.log_clients_result(eval_results, phase='post_agg', event='local_eval')

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

            self.result_manager.log_clients_result(train_result, phase='post_train', event='local_train')
            self.result_manager.log_clients_result(eval_result, phase='post_train', event='local_eval')

            self.result_manager.update_round_and_flush(curr_round)

    def run_centralized_simulation(self):
        # TODO: Just initiate one client and pass all the data to that client
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
            client: BaseClient = client_partial(id_seed=idx, dataset=datasets, model=model, res_man=self.result_manager)
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
