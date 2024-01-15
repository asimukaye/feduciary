import os
import glob
import typing as t
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import time
import logging
import random
import numpy as np
import torch
from functools import partial
import wandb
from torch.nn import Module
from torch.backends import cudnn, mps
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
import flwr as fl

#FIXME: Eventually everything should also work with the base server and base server. Merge once the first logical tests pass for flower
# from src.server.baseserver import BaseServer
from src.server.baseflowerserver import BaseFlowerServer
# from src.server.fedstdevserver import FedstdevClient, FedstdevOptimizer
# from src.client.baseclient import BaseClient, model_eval_helper
from src.client.baseflowerclient import BaseFlowerClient, model_eval_helper

from src.data import load_vision_dataset
from src.split import get_client_datasets, NoisySubset, LabelFlippedSubset
from src.config import Config, SimConfig, ClientSchema
from src.common.utils  import log_tqdm, log_instance, generate_client_ids
from src.results.postprocess import post_process
from src.models.model import init_model
from src.results.resultmanager import ResultManager
from src.metrics.metricmanager import MetricManager
import src.common.typing as fed_t

logger = logging.getLogger('SIMULATOR')

# @dataclass
# class SimIns:
#     cfg: Config
#     train_set: 
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info(f'[SEED] Simulator global seed is set to: {seed}!')


def _create_client(cid: str, datasets, model: Module, client_cfg: ClientSchema) -> BaseFlowerClient:

    client_partial: partial = instantiate(client_cfg)
    # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    # Always deepcopy the model
    client: BaseFlowerClient = client_partial(client_id=cid, dataset=datasets, model = deepcopy(model))
    return client


def create_clients(all_client_ids, client_datasets, model_instance, client_cfg: ClientSchema) -> dict[str, BaseFlowerClient]:

    clients = {}
    for cid, datasets in log_tqdm(zip(all_client_ids,client_datasets), logger=logger, desc=f' creating clients '):
        # client_id = f'{idx:04}' # potential to convert to a unique hash
        client_obj = _create_client(cid, datasets, model_instance, client_cfg)
        clients[cid] = client_obj
    return clients



def make_checkpoint_dirs(has_server: bool, client_ids=[]):
    if has_server:
        os.makedirs('server_ckpts', exist_ok = True)
    
    os.makedirs('client_ckpts', exist_ok = True)
    
    for cid in client_ids:
        os.makedirs(f'client_ckpts/{cid}', exist_ok = True)

def find_checkpoint()-> tuple[str, dict]:
    server_ckpts = sorted(glob.glob('server_ckpts/server_ckpt_*'))
    client_ckpts = {}

    for dir in os.listdir('client_ckpts/'):
        files = sorted(os.listdir(f'client_ckpts/{dir}'))
        if files:
            client_ckpts[dir] = f'client_ckpts/{dir}/{files[-1]}'
            logger.info(f'------ Found client {dir} checkpoint: {client_ckpts[dir]} ------')
        
    if server_ckpts or client_ckpts:
        if server_ckpts:
            logger.info(f'------ Found server checkpoint: {server_ckpts[-1]} ------')
            return server_ckpts[-1], client_ckpts
        else:
            return '', client_ckpts
    else:
        logger.debug('------------ No checkpoints found. Starting afresh ------------')
        return '', {}
    

def save_checkpoints(server: BaseFlowerServer, clients: dict[str, BaseFlowerClient]):
    if server:
        server.save_checkpoint()
    if clients:
        for client in clients.values():
            client.save_checkpoint()

# TODO: Test method to load client checkpoints
# def load_state(self, server_ckpt_path: str, client_ckpts: dict):
#     if server_ckpt_path:
#         self.server.load_checkpoint(server_ckpt_path)
#         self._round = self.server._round
#     if client_ckpts:
#         for cid, ckpt in client_ckpts.items():
#             self.clients[cid].load_checkpoint(ckpt)
#             if self._round == 0:
    

def init_dataset_and_model(cfg: Config) -> tuple[Subset, Subset, Module]:
    '''Initialize the dataset and the model here'''
    # NOTE: THIS FUNCTION MODIFIES THE RANDOM NUMBER SEEDS
    # NOTE: THe model spec is being modified in place here.
    # TODO: Generalize this logic for all datasets
    test_set, train_set, dataset_model_spec  = load_vision_dataset(cfg.dataset)
    # client_sets = get_client_datasets(cfg.dataset.split_conf, train_set)

    model_spec = cfg.model.model_spec
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

    cfg.model.model_spec = model_spec

    # model_instance: Module = instantiate(cfg.model.model_spec)
    model_instance = init_model(cfg.model)
    return test_set, train_set, model_instance


def run_flower_simulation(cfg: Config,
                        train_set: Subset,
                        test_set: Subset,
                        model: Module):
    
    
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    make_checkpoint_dirs(has_server=True, client_ids=all_client_ids)
    clients: dict[str, BaseFlowerClient] = dict()

    # server_dataset, train_set, model = init_dataset_and_model(cfg)
    server_dataset = test_set
    client_datasets =  get_client_datasets(cfg.dataset.split_conf, train_set)

    client_datasets_map = {}
    for cid, dataset in zip(all_client_ids,client_datasets):
        client_datasets_map[cid] = dataset

    if torch.cuda.is_available():
        cfg.client.train_cfg.device = 'cuda:0'
        cfg.server.train_cfg.device = 'cuda:0'
    elif mps.is_available():
        cfg.client.train_cfg.device = 'mps'
        cfg.server.train_cfg.device = 'mps'
    else:
        cfg.client.train_cfg.device = 'cpu'
        cfg.server.train_cfg.device = 'cpu'

    result_manager = ResultManager(cfg.simulator, logger=logger)

    strategy = instantiate(cfg.strategy, model)

    server_partial = instantiate(cfg.server)
    server: BaseFlowerServer = server_partial(model=model, dataset=server_dataset, clients= clients, strategy=strategy, result_manager=result_manager)


    def _client_fn(cid: str):
        client_partial: partial = instantiate(cfg.client)
        _model = deepcopy(model)
        _datasets = client_datasets_map[cid]

        return client_partial(client_id=cid,
                dataset=_datasets, model=_model)
    
    # Flower simulation arguments
    # runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))}}
    runtime_env = {}
    # runtime_env['working_dir'] = "/home/asim.ukaye/fed_learning/feduciary/"
    runtime_env['working_dir'] = os.getcwd()


    fl.simulation.start_simulation(
        strategy=server,
        client_fn=_client_fn,
        clients_ids= all_client_ids,
        config=fl.server.ServerConfig(num_rounds=cfg.simulator.num_rounds),
        ray_init_args = {'runtime_env': runtime_env},
        client_resources=cfg.simulator.flwr_resources,
    )

    
def run_federated_simulation(cfg: Config,
                             train_set: Subset,
                             test_set: Subset,
                             model_instance: Module
                             ):
    '''Runs the simulation in federated mode'''
    
    # model_instance: Module = instantiate(cfg.model.model_spec)
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    make_checkpoint_dirs(has_server=True, client_ids=all_client_ids)

    server_partial: partial = instantiate(cfg.server)
    clients: dict[str, BaseFlowerClient] = dict()
    
    result_manager = ResultManager(cfg.simulator, logger=logger)
    
    #  Server gets the test set
    server_dataset = test_set
    # Clients get the splits of the train set with an inbuilt test set
    client_datasets =  get_client_datasets(cfg.dataset.split_conf, train_set)

    # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    clients = create_clients(all_client_ids, client_datasets, model_instance, cfg.client)

    # HACK: Fixing batch size for imbalanced client case
    # FIXME: Formalize this later
    if cfg.dataset.split_conf.split_type == 'one_imbalanced_client':
        clients['0000'].train_cfg.batch_size = int(clients['0000'].train_cfg.batch_size/2)
        for cid, cl in clients.items():
            logger.debug(f'[BATCH SIZES:] CID: {cid}, batch size: {cl.train_cfg.batch_size}')

    # all_client_ids = list(clients.keys())
    # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary
    strategy = instantiate(cfg.strategy, model=model_instance)

    server: BaseFlowerServer = server_partial(model=model_instance, strategy=strategy, dataset=server_dataset, clients= clients, result_manager=result_manager)


    _round = 0
    # clients = _create_clients(client_datasets)
    # server.initialize(clients, )

    for curr_round in range(_round, cfg.simulator.num_rounds):
        logger.info(f'-------- Round: {curr_round} --------\n')
        # wandb.log({'round': curr_round})
        loop_start = time.time()
        _round = curr_round
        ## update round indicator
        server._round = curr_round

        ## update after sampling clients randomly
        # TODO: Add availability logic

        update_ids = server.update(all_client_ids)

        ## evaluate on clients not sampled (for measuring generalization performance)
        if curr_round % cfg.simulator.eval_every == 0:
            # Can have specific evaluations later
            eval_ids = all_client_ids
            server.server_eval()
            eval_ids = server.local_eval(all_client_ids)

        # if curr_round % sim_cfg.checkpoint_every == 0:
        #     save_checkpoints()
        
        # This is weird, needs some rearch
        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')

    final_result = result_manager.finalize()
    return final_result

def run_centralized_simulation(cfg: Config,
                               train_set: Subset,
                               test_set: Subset,
                               model_instance: Module):
                # Reusing clients dictionary to repurpose existing code
    clients: dict[str, BaseFlowerClient] = defaultdict()

    make_checkpoint_dirs(has_server=False, client_ids=['centralized'])
    # TODO: Formalize this later outside of simulator
    # Modify the dataset here:
    split_conf = cfg.dataset.split_conf
    logger.info(f'[DATA_SPLIT] Simulated dataset split : `{split_conf.split_type}`')

    if split_conf.split_type == 'one_noisy_client':
        train_set =  NoisySubset(train_set, split_conf.noise.mu, split_conf.noise.sigma)
    elif cfg.dataset.split_conf.split_type == 'one_label_flipped_client':
        train_set = LabelFlippedSubset(train_set, split_conf.noise.flip_percent)

    clients['centralized'] = _create_client('centralized', (train_set, test_set), model_instance, cfg.client)

    result_manager = ResultManager(cfg.simulator, logger=logger)

        # FIXME: Clean this part for general centralized runs
    # trainer: FedstdevClient
    trainer: FedstdevClient = clients['centralized']

    # Additional code for current usecase. Factorize later
    strat_cfg: FedstdevServerConfig = cfg.server.cfg

    param_dims = {p_key: np.prod(param.size()) for p_key, param in model_instance.named_parameters()} #type: ignore
    param_keys = param_dims.keys()
    betas = {param: beta for param, beta in zip(param_keys, strat_cfg.betas)}
    _round = 0

    for curr_round in range(_round, cfg.simulator.num_rounds):
        logger.info(f'-------- Round: {curr_round} --------\n')
        # wandb.log({'_round': curr_round})
        loop_start = time.time()
        _round = curr_round
        trainer._round = curr_round

        train_result = trainer.train()
        result_manager.log_general_result(train_result, 'post_train', 'sim', 'central_train')
        
        #  Logging and evaluating specific to fedstdev
        params = trainer.upload()
        result_manager.log_parameters(params, 'post_train', 'sim', verbose=True)

        param_stdev = trainer._get_parameter_std_dev()
        grad_stdev = trainer._get_gradients_std_dev()
        grad_mu = trainer._get_gradients_average()


        param_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(param_stdev, params)

        grad_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(grad_stdev, grad_mu)
            

        omegas = FedstdevOptimizer._compute_scaled_weights(betas, std_dict=grad_sigma_by_mu)

        
        param_std_dct = result_manager.log_parameters(param_stdev, 'post_train', 'sim',metric='param_std', verbose=True)
        param_sbm_dct = result_manager.log_parameters(param_sigma_by_mu, 'post_train', 'sim', metric='sigma_by_mu', verbose=True)
        grad_mu_dct = result_manager.log_parameters(grad_mu, 'post_train', 'sim',metric='grad_mu')
        grad_std_dct = result_manager.log_parameters(grad_stdev, 'post_train', 'sim',metric='grad_std')
        grad_sbm_dct = result_manager.log_parameters(grad_sigma_by_mu, 'post_train', 'sim', metric='grad_sigma_by_mu')

        result_manager.log_general_metric(FedstdevOptimizer.get_dict_avg(omegas, param_dims), f'omegas', 'sim', 'post_train')

        # Logging code ends here

        eval_result = trainer.eval()
        result_manager.log_general_result(eval_result, 'post_eval', 'sim', 'central_eval')

        # if curr_round % sim_cfg.checkpoint_every == 0:
        #     save_checkpoints()
        
        # This is weird, needs some rearch
        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')
    final_result = result_manager.finalize()
    return final_result


def run_standalone_simulation(cfg: Config,
                              train_set: Subset,
                              model_instance: Module):
    
    clients: dict[str, BaseFlowerClient] = defaultdict()
    # Clients get the splits of the train set with an inbuilt test set
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)

    make_checkpoint_dirs(has_server=False, client_ids=all_client_ids)

    client_datasets = get_client_datasets(cfg.dataset.split_conf, train_set)

    result_manager = ResultManager(cfg.simulator, logger=logger)

    clients = create_clients(all_client_ids, client_datasets, model_instance, cfg.client)
    
    _round = 0
    for curr_round in range(_round, cfg.simulator.num_rounds):
        train_result = {}
        eval_result = {}

        for cid, client in clients.items():
            logger.info(f'-------- Client: {cid} --------\n')
            client._round = curr_round
            train_result[cid] = client.train()
            eval_result[cid] = client.eval()
        # if curr_round % sim_cfg.checkpoint_every == 0:
        #     save_checkpoints()

        result_manager.log_clients_result(train_result, phase='post_train', event='local_train')
        result_manager.log_clients_result(eval_result, phase='post_train', event='local_eval')

        result_manager.flush_and_update_round(curr_round)
    
    final_result = result_manager.finalize()
    return final_result


class Simulator:
    """Simulator orchestrating the whole process of federated learning.
    """
    # TODO: Split into federated, standalone, and centralized simulator
    def __init__(self, cfg: Config):
        self.start_time = time.time()
        self._round: int = 0
        self.cfg: Config = cfg
        self.sim_cfg: SimConfig = cfg.simulator
   
        if self.sim_cfg.use_wandb:
            wandb.init(project='fed_ml', job_type=cfg.mode,
                        config=asdict(cfg), resume=True, notes=cfg.desc)

        logger.info(f'[SIM MODE] : {self.sim_cfg.mode}')
        logger.info(f'[SERVER] : {self.cfg.server._target_.split(".")[-1]}')
        logger.info(f'[STRATEGY] : {self.cfg.strategy._target_.split(".")[-1]}')
        logger.info(f'[CLIENT] : {self.cfg.client._target_.split(".")[-1]}')

        logger.info(f'[NUM ROUNDS] : {self.sim_cfg.num_rounds}')

        # self.num_clients = cfg.simulator.num_clients
        # self.all_client_ids = generate_client_ids(cfg.simulator.num_clients)

        set_seed(cfg.simulator.seed)
        # self.algo = cfg.server.algorithm.name
        # Remove calls like thes to make things more testable

        self.test_set, self.train_set, self.model_instance = init_dataset_and_model(cfg=cfg)

        # self.metric_manager = MetricManager(cfg.client.cfg.metric_cfg, self._round, actor='simulator')

        # self.result_manager = ResultManager(cfg.simulator, logger=logger)

        # Till here can be factorized
        self.server = None

        # self.set_fn_overloads_for_mode()

        # self.init_sim()

        # TODO: consolidate checkpointing and resuming logic systematically

        # self.make_checkpoint_dirs()
        self.is_resumed =False
        
        # server_ckpt, client_ckpts = self.find_checkpoint()

        # # FIXME: need to fix resume logic
        # if self.is_resumed:
        #     logger.info('------------ Resuming training ------------')
        #     self.load_state(server_ckpt, client_ckpts)
        
        logger.debug(f'Init time: {time.time() - self.start_time} seconds')

    def run_simulation(self):
        match self.sim_cfg.mode:
            case 'federated':
                run_federated_simulation(cfg = self.cfg,
                                         train_set=self.train_set,
                                         test_set=self.test_set,
                                         model_instance=self.model_instance)
            case 'standalone':
                run_standalone_simulation(cfg=self.cfg,
                                          train_set=self.train_set,
                                          model_instance=self.model_instance)            
            case 'centralized':
                run_centralized_simulation(cfg = self.cfg,
                                           train_set=self.train_set,
                                           test_set=self.test_set,
                                           model_instance=self.model_instance)
            case 'flower':
                run_flower_simulation(cfg = self.cfg,
                                    train_set=self.train_set,
                                    test_set=self.test_set,
                                    model=self.model_instance)
            case _:
                raise AssertionError(f'Mode: {self.sim_cfg.mode} is not implemented')


    # def set_fn_overloads_for_mode(self):
    #     match self.sim_cfg.mode:
    #         case 'federated':
    #             self.init_sim = self.init_federated_mode
    #             self.run_simulation = self.run_federated_simulation
    #         case 'standalone':
    #             self.init_sim = self.init_standalone_mode
    #             self.run_simulation = self.run_standalone_simulation            
    #         case 'centralized':
    #             self.init_sim = self.init_centralized_mode
    #             self.run_simulation = self.run_centralized_simulation
    #         case 'flower':
    #             self.init_sim = init_flower_sim(cfg=self.cfg)
    #             self.run_simulation = run_flower_simulation(cfg)
    #         case _:
    #             raise AssertionError(f'Mode: {self.sim_cfg.mode} is not implemented')

    # def init_federated_mode(self):
    #     # model_instance: Module = instantiate(self.cfg.model.model_spec)

    #     server_partial: partial = instantiate(self.cfg.server)
    #     self.clients: dict[str, BaseFlowerClient] = dict()
        
    #     #  Server gets the test set
    #     self.server_dataset = self.test_set
    #     # Clients get the splits of the train set with an inbuilt test set
    #     self.client_datasets =  get_client_datasets(self.cfg.dataset.split_conf, self.train_set)


    #     # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    #     self.clients = self._create_clients(self.client_datasets)

    #     # HACK: Fixing batch size for imbalanced client case
    #     # FIXME: Formalize this later
    #     if self.cfg.dataset.split_conf.split_type == 'one_imbalanced_client':
    #         self.clients['0000'].cfg.batch_size = int(self.clients['0000'].cfg.batch_size/2)
    #         for cid, cl in self.clients.items():
    #             logger.debug(f'[BATCH SIZES:] CID: {cid}, batch size: {cl.cfg.batch_size}')

    #     # self.all_client_ids = list(self.clients.keys())
    #     # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary
    #     strategy = instantiate(self.cfg.strategy, model=self.model_instance)
    #     self.server: BaseFlowerServer = server_partial(model=self.model_instance, strategy=strategy, dataset=self.server_dataset, clients= self.clients, result_manager=self.result_manager)

    # def init_standalone_mode(self):
    #     self.clients: dict[str, BaseFlowerClient] = defaultdict(BaseFlowerClient)

    #     # Clients get the splits of the train set with an inbuilt test set
    #     self.client_datasets =  get_client_datasets(self.cfg.dataset.split_conf, self.train_set)


    #     self.clients = self._create_clients(self.client_datasets)

    # def init_centralized_mode(self):
    #     # Reusing clients dictionary to repurpose existing code
    #     self.clients: dict[str, BaseFlowerClient] = defaultdict(BaseFlowerClient)

    #     # TODO: Formalize this later outside of simulator
    #     # Modify the dataset here:
    #     split_conf = self.cfg.dataset.split_conf
    #     logger.info(f'[DATA_SPLIT] Simulated dataset split : `{split_conf.split_type}`')

    #     if split_conf.split_type == 'one_noisy_client':
    #         self.train_set =  NoisySubset(self.train_set, split_conf.noise.mu, split_conf.noise.sigma)
    #     elif self.cfg.dataset.split_conf.split_type == 'one_label_flipped_client':
    #         self.train_set = LabelFlippedSubset(self.train_set, split_conf.noise.flip_percent)

    #     self.clients['centralized']: BaseFlowerClient = self.__create_client('centralized', (self.train_set, self.test_set), self.model_instance)

        

    # def make_checkpoint_dirs(self):
    #     os.makedirs('server_ckpts', exist_ok = True)
    #     for cid in self.all_client_ids:
    #         os.makedirs(f'client_ckpts/{cid}', exist_ok = True)

    # def find_checkpoint(self)-> tuple[str, dict]:
    #     server_ckpts = sorted(glob.glob('server_ckpts/server_ckpt_*'))
    #     client_ckpts = {}

    #     for dir in os.listdir('client_ckpts/'):
    #         files = sorted(os.listdir(f'client_ckpts/{dir}'))
    #         if files:
    #             client_ckpts[dir] = f'client_ckpts/{dir}/{files[-1]}'
    #             logger.info(f'------ Found client {dir} checkpoint: {client_ckpts[dir]} ------')
            
    #     if server_ckpts or client_ckpts:
    #         self.is_resumed = True
    #         if server_ckpts:
    #             logger.info(f'------ Found server checkpoint: {server_ckpts[-1]} ------')
    #             return server_ckpts[-1], client_ckpts
    #         else:
    #             return '', client_ckpts
    #     else:
    #         logger.debug('------------ No checkpoints found. Starting afresh ------------')
    #         self.is_resumed = False
    #         return '', {}


    
    # def save_checkpoints(self):
    #     if self.server:
    #         self.server.save_checkpoint()
    #     if self.clients:
    #         for client in self.clients.values():
    #             client.save_checkpoint()

    # # TODO: Test method to load client checkpoints
    # def load_state(self, server_ckpt_path: str, client_ckpts: dict):
    #     if server_ckpt_path:
    #         self.server.load_checkpoint(server_ckpt_path)
    #         self._round = self.server._round
    #     if client_ckpts:
    #         for cid, ckpt in client_ckpts.items():
    #             self.clients[cid].load_checkpoint(ckpt)
    #             if self._round == 0:
    #                 self._round = self.clients[cid]._round

    # def __create_client(self, cid: str, datasets, model: Module) -> BaseFlowerClient:

    #     client_partial: partial = instantiate(self.cfg.client)
    #     # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    #     # Always deepcopy the model
    #     client: BaseFlowerClient = client_partial(client_id=cid, dataset=datasets, model=deepcopy(model))
    #     return client
    
    # # @log_instance(attrs=['_round', 'num_clients'], m_logger=logger)
    # def _create_clients(self, client_datasets) -> dict[str, BaseFlowerClient]:

    #     clients = {}
    #     for cid, datasets in log_tqdm(zip(self.all_client_ids,client_datasets), logger=logger, desc=f'[Round: {self._round:03}] creating clients '):
    #         # client_id = f'{idx:04}' # potential to convert to a unique hash
    #         client_obj = self.__create_client(cid, datasets, self.model_instance)
    #         clients[cid] = client_obj
    #     return clients

    # def central_evaluate_clients(self, cids: list[str]):
    #     '''Evaluate the clients on servers holdout set'''
    #     # NOTE: this is useless if the central evaluation is done after the last aggregation
    #     test_loader = DataLoader(dataset=self.server_dataset, batch_size=self.cfg.client.cfg.batch_size)
    #     eval_result = {}
    #     # FIXME: Might need to instantiate globally elsewhere
    #     client_cfg_inst = instantiate(self.cfg.client.cfg)
    #     for cid in cids:
    #         eval_result[cid] = model_eval_helper(self.clients[cid].model, test_loader, client_cfg_inst, self.metric_manager, self._round)
        
    #     self.result_manager.log_clients_result(eval_result, phase='post_agg', event='central_eval')
        
    # def run_standalone_simulation(self):
    #     # 
    #     for curr_round in range(self._round, self.sim_cfg.num_rounds + 1):
    #         train_result = {}
    #         eval_result = {}

    #         for cid, client in self.clients.items():
    #             logger.info(f'-------- Client: {cid} --------\n')
    #             client._round = curr_round
    #             train_result[cid] = client.train()
    #             eval_result[cid] = client.eval()
    #         if curr_round % self.sim_cfg.checkpoint_every == 0:
    #             self.save_checkpoints()

    #         self.result_manager.log_clients_result(train_result, phase='post_train', event='local_train')
    #         self.result_manager.log_clients_result(eval_result, phase='post_train', event='local_eval')

    #         self.result_manager.flush_and_update_round(curr_round)

    # def run_centralized_simulation(self):
    #     # FIXME: Clean this part for general centralized runs
    #     # self.trainer: FedstdevClient
    #     trainer: FedstdevClient = self.clients['centralized']

    #     # Additional code for current usecase. Factorize later
    #     strat_cfg: FedstdevServerConfig = self.cfg.server.cfg

    #     param_dims = {p_key: np.prod(param.size()) for p_key, param in self.model_instance.named_parameters()} #type: ignore
    #     param_keys = param_dims.keys()
    #     betas = {param: beta for param, beta in zip(param_keys, strat_cfg.betas)}

    #     for curr_round in range(self._round, self.sim_cfg.num_rounds + 1):
    #         logger.info(f'-------- Round: {curr_round} --------\n')
    #         # wandb.log({'_round': curr_round})
    #         loop_start = time.time()
    #         self._round = curr_round
    #         trainer._round = curr_round


    #         train_result = trainer.train()
    #         self.result_manager.log_general_result(train_result, 'post_train', 'sim', 'central_train')
            
    #         #  Logging and evaluating specific to fedstdev
    #         params = trainer.upload()
    #         self.result_manager.log_parameters(params, 'post_train', 'sim', verbose=True)

    #         param_stdev = trainer.get_parameter_std_dev()
    #         grad_stdev = trainer.get_gradients_std_dev()
    #         grad_mu = trainer.get_gradients_average()


    #         param_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(param_stdev, params)

    #         grad_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(grad_stdev, grad_mu)

    #         # for (name, grad_avg), grad_std, grad_sbm in zip(grad_mu.items(), grad_stdev.values(), grad_sigma_by_mu.values()):
    #             # ic()
    #             # ic(name)
    #             # ic(grad_avg.shape)
    #             # ic(grad_std.shape)
    #             # ic(grad_sbm.shape)

    #             # ic(grad_std.view(-1)[0])
    #             # ic(grad_avg.view(-1)[0])
    #             # ic(grad_sbm.view(-1)[0])
    #             # ic(grad_std.view(-1)[0]/grad_avg.view(-1)[0])


    #             # ic(grad_std.abs().mean().item())
    #             # ic(grad_avg.abs().mean().item())
    #             # ic(grad_sbm.abs().mean().item())
                

    #         omegas = FedstdevOptimizer._compute_scaled_weights(betas, std_dict=grad_sigma_by_mu)

            
    #         param_std_dct = self.result_manager.log_parameters(param_stdev, 'post_train', 'sim',metric='param_std', verbose=True)
    #         param_sbm_dct = self.result_manager.log_parameters(param_sigma_by_mu, 'post_train', 'sim', metric='sigma_by_mu', verbose=True)
    #         grad_mu_dct = self.result_manager.log_parameters(grad_mu, 'post_train', 'sim',metric='grad_mu')
    #         grad_std_dct = self.result_manager.log_parameters(grad_stdev, 'post_train', 'sim',metric='grad_std')
    #         grad_sbm_dct = self.result_manager.log_parameters(grad_sigma_by_mu, 'post_train', 'sim', metric='grad_sigma_by_mu')

    #         # ic(grad_sbm_dct['wtd_avg'])
    #         # ic(grad_std_dct['wtd_avg'])

    #         # ic(grad_std_dct['wtd_avg']/grad_mu_dct['wtd_avg'])


    #         # ic(grad_sbm_dct['avg'])
    #         # ic(grad_std_dct['avg'])

    #         # ic(grad_std_dct['avg']/grad_mu_dct['avg'])
    #         self.result_manager.log_general_metric(FedstdevOptimizer.get_dict_avg(omegas, param_dims), f'omegas', 'sim', 'post_train')

    #         # Logging code ends here

    #         eval_result = trainer.eval()
    #         self.result_manager.log_general_result(eval_result, 'post_eval', 'sim', 'central_eval')

    #         if curr_round % self.sim_cfg.checkpoint_every == 0:
    #             self.save_checkpoints()
            
    #         # This is weird, needs some rearch
    #         self.result_manager.flush_and_update_round(curr_round)

    #         loop_end = time.time() - loop_start
    #         logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')


    # def run_federated_simulation(self):
    #     '''Runs the simulation in federated mode'''
    #     # self.clients = self._create_clients(self.client_datasets)
    #     # self.server.initialize(clients, )

    #     for curr_round in range(self._round, self.sim_cfg.num_rounds +1):
    #         logger.info(f'-------- Round: {curr_round} --------\n')
    #         # wandb.log({'round': curr_round})
    #         loop_start = time.time()
    #         self._round = curr_round
    #         ## update round indicator
    #         self.server._round = curr_round

    #         ## update after sampling clients randomly
    #         # TODO: Add availability logic

    #         update_ids = self.server.update(self.all_client_ids)

    #         ## evaluate on clients not sampled (for measuring generalization performance)
    #         if curr_round % self.cfg.server.cfg.eval_every == 0:
    #             # Can have specific evaluations later
    #             eval_ids = self.all_client_ids
    #             self.server.server_eval()
    #             eval_ids = self.server.local_eval(self.all_client_ids)

    #         if curr_round % self.sim_cfg.checkpoint_every == 0:
    #             self.save_checkpoints()
            
    #         # This is weird, needs some rearch
    #         self.result_manager.flush_and_update_round(curr_round)

    #         loop_end = time.time() - loop_start
    #         logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')

    
    def finalize(self):
        # if self.server:
        #     self.server.finalize()
        total_time= time.time() - self.start_time
        logger.info(f'Total runtime: {total_time} seconds')

        # FIXME: Post process is broken
        # post_process(self.cfg, final_result, total_time=total_time)

        # del self.clients
        # del self.server
        logger.info('Closing Feduciary Simulator')
