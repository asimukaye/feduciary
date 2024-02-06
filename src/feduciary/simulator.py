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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from hydra.utils import instantiate
import flwr as fl

# from feduciary.server.baseserver import BaseServer
from feduciary.server.baseflowerserver import BaseFlowerServer
# from feduciary.server.fedstdevserver import FedstdevClient, FedstdevOptimizer
from feduciary.client.abcclient import simple_evaluator, simple_trainer
from feduciary.client.baseflowerclient import BaseFlowerClient

from feduciary.data import load_vision_dataset, load_raw_dataset
from feduciary.split import get_client_datasets, NoisySubset, LabelFlippedSubset
from feduciary.config import Config, SimConfig, ClientSchema
from feduciary.common.utils  import log_tqdm, log_instance, generate_client_ids
from feduciary.results.postprocess import post_process
from feduciary.models.model import init_model
from feduciary.results.resultmanager import ResultManager
from feduciary.metrics.metricmanager import MetricManager
import feduciary.common.typing as fed_t

logger = logging.getLogger('SIMULATOR')


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


def _create_client(cid: str, datasets: fed_t.DatasetPair_t, model: Module, client_cfg: ClientSchema) -> BaseFlowerClient:

    client_partial: partial = instantiate(client_cfg)
    # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    # Always deepcopy the model
    client: BaseFlowerClient = client_partial(client_id=cid, dataset=datasets, model = deepcopy(model))
    return client


def create_clients(all_client_ids, client_datasets, model_instance, client_cfg: ClientSchema) -> dict[str, BaseFlowerClient]:

    clients = {}
    for cid, datasets in log_tqdm(zip(all_client_ids, client_datasets), logger=logger, desc=f'creating clients '):
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
    

def naive_aggregator(client_params: fed_t.ClientParams_t) -> fed_t.ActorParams_t:
    '''Naive averaging of client parameters'''
    server_params = {}
    client_ids = list(client_params.keys())
    param_keys = client_params[client_ids[0]].keys()
    for key in param_keys:
        server_params[key] = torch.stack([client[key].data for client in client_params.values()]).mean(dim=0)
    return server_params

def save_checkpoints(server: BaseFlowerServer, clients: dict[str, BaseFlowerClient]):
    if server:
        server.save_checkpoint()
    if clients:
        for client in clients.values():
            client.save_checkpoint()



def init_dataset_and_model(cfg: Config) -> tuple[fed_t.ClientDatasets_t, Subset, Module]:
    '''Initialize the dataset and the model here'''
    # NOTE: THIS FUNCTION MODIFIES THE RANDOM NUMBER SEEDS
    # NOTE: This function modifies the config objec insitu

    train_set, test_set, dataset_model_spec  = load_raw_dataset(cfg.dataset)
    client_sets = get_client_datasets(cfg.dataset.split_conf, train_set, test_set)

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

    # Set the value of n_iters for the client
    per_client_set_size = len(train_set)//cfg.simulator.num_clients
    cfg.client.cfg.n_iters = per_client_set_size//cfg.client.train_cfg.batch_size + (1 if per_client_set_size%cfg.client.train_cfg.batch_size else 0)
    logger.debug(f'[DATA_SPLIT] N iters : `{cfg.client.cfg.n_iters}`')
    logger.debug(f'[DATA_SPLIT] batch size : `{cfg.client.train_cfg.batch_size}`')


    return client_sets, test_set, model_instance


def run_flower_simulation(cfg: Config,
                        client_datasets: fed_t.ClientDatasets_t,
                        server_dataset: Subset,
                        model: Module):
    
    
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    make_checkpoint_dirs(has_server=True, client_ids=all_client_ids)
    clients: dict[str, BaseFlowerClient] = dict()

    client_datasets_map = {}
    for cid, dataset in zip(all_client_ids, client_datasets):
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

    strategy = instantiate(cfg.strategy, model=model, res_man=result_manager)

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
                             client_datasets: fed_t.ClientDatasets_t,
                             server_dataset: Subset,
                             model_instance: Module
                             ):
    '''Runs the simulation in federated mode'''
    
    # model_instance: Module = instantiate(cfg.model.model_spec)
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    make_checkpoint_dirs(has_server=True, client_ids=all_client_ids)

    server_partial: partial = instantiate(cfg.server)
    clients: dict[str, BaseFlowerClient] = dict()
    
    # strategy = instantiate(cfg.strategy, model=model_instance)


    result_manager = ResultManager(cfg.simulator, logger=logger)
    strategy = instantiate(cfg.strategy, model=model_instance, res_man=result_manager)
    
    #  Server gets the test set
    # server_dataset = test_set
    # Clients get the splits of the train set with an inbuilt test set
    # client_datasets =  get_client_datasets(cfg.dataset.split_conf, train_set, test_set, match_train_distribution=False)

    # NOTE:IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    clients = create_clients(all_client_ids, client_datasets, model_instance, cfg.client)

    # NOTE: later, consider making a copy of client to avoid simultaneous edits to clients dictionary

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
                             client_datasets: fed_t.ClientDatasets_t,
                             server_dataset: Subset,
                               model: Module):

    # make_checkpoint_dirs(has_server=False, client_ids=['centralized'])

    pooled_train_set = ConcatDataset([train_set for train_set, _ in client_datasets])
    pooled_test_set = ConcatDataset([test_set for _, test_set in client_datasets])

    cfg.train_cfg.criterion = instantiate(cfg.train_cfg.criterion)
    cfg.train_cfg.optimizer = instantiate(cfg.train_cfg.optimizer)(model.parameters())

    train_loader = DataLoader(dataset=pooled_train_set,
                              batch_size=cfg.train_cfg.batch_size, shuffle=True)
    
    test_loader = DataLoader(dataset=server_dataset,
                                       batch_size=cfg.train_cfg.eval_batch_size, shuffle=False)
    
    result_manager = ResultManager(cfg.simulator, logger=logger)
    metric_manager = MetricManager(cfg.train_cfg.metric_cfg, 0, actor='simulator')

    for curr_round in range(cfg.simulator.num_rounds):
        logger.info(f'-------- Round: {curr_round} --------\n')

        loop_start = time.time()

        train_result = simple_trainer(model, train_loader, cfg.train_cfg, metric_manager, curr_round)

        result_manager.log_general_result(train_result, 'post_train', 'sim', 'central_train')
   
        params = model.state_dict()
        result_manager.log_parameters(params, 'post_train', 'sim', verbose=True)


        eval_result = simple_evaluator(model, test_loader, cfg.train_cfg, metric_manager, curr_round)
        result_manager.log_general_result(eval_result, 'post_eval', 'sim', 'central_eval')

        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')
    final_result = result_manager.finalize()
    return final_result

#FIXME: This is a temporary function to test the single client simulation
def run_single_client(cfg: Config,
                      client_datasets: fed_t.ClientDatasets_t,
                      server_dataset: Subset,
                      model: Module):
    # Reusing clients dictionary to repurpose existing code
    clients: dict[str, BaseFlowerClient] = defaultdict()

    make_checkpoint_dirs(has_server=False, client_ids=['single_client'])

     # TODO: Formalize this later outside of simulator
    # Modify the dataset here:
    split_conf = cfg.dataset.split_conf
    logger.info(f'[DATA_SPLIT] Simulated dataset split : `{split_conf.split_type}`')

    # if split_conf.split_type == 'one_noisy_client':
    #     train_set =  NoisySubset(train_set, split_conf.noise.mu, split_conf.noise.sigma)
    # elif cfg.dataset.split_conf.split_type == 'one_label_flipped_client':
    #     train_set = LabelFlippedSubset(train_set, split_conf.noise.flip_percent)

    # clients['centralized'] = _create_client('centralized', (train_set, test_set),  model, cfg.client)

    result_manager = ResultManager(cfg.simulator, logger=logger)

    #     # FIXME: Clean this part for general centralized runs
    # # trainer: FedstdevClient
    # trainer: FedstdevClient = clients['centralized']

    # # Additional code for current usecase. Factorize later
    # strat_cfg: FedstdevServerConfig = cfg.server.cfg

    # param_dims = {p_key: np.prod(param.size()) for p_key, param in model_instance.named_parameters()} #type: ignore
    # param_keys = param_dims.keys()
    # betas = {param: beta for param, beta in zip(param_keys, strat_cfg.betas)}
    # _round = 0

    for curr_round in range(cfg.simulator.num_rounds):
        logger.info(f'-------- Round: {curr_round} --------\n')
        loop_start = time.time()
    #     _round = curr_round
    #     trainer._round = curr_round

    #     train_result = trainer.train()
    #     result_manager.log_general_result(train_result, 'post_train', 'sim', 'central_train')
        
    #     #  Logging and evaluating specific to fedstdev
    #     params = trainer.upload()
    #     result_manager.log_parameters(params, 'post_train', 'sim', verbose=True)

    #     param_stdev = trainer._get_parameter_std_dev()
    #     grad_stdev = trainer._get_gradients_std_dev()
    #     grad_mu = trainer._get_gradients_average()


    #     param_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(param_stdev, params)

    #     grad_sigma_by_mu = FedstdevOptimizer._compute_sigma_by_mu(grad_stdev, grad_mu)
            

    #     omegas = FedstdevOptimizer._compute_scaled_weights(betas, std_dict=grad_sigma_by_mu)

        
    #     param_std_dct = result_manager.log_parameters(param_stdev, 'post_train', 'sim',metric='param_std', verbose=True)
    #     param_sbm_dct = result_manager.log_parameters(param_sigma_by_mu, 'post_train', 'sim', metric='sigma_by_mu', verbose=True)
    #     grad_mu_dct = result_manager.log_parameters(grad_mu, 'post_train', 'sim',metric='grad_mu')
    #     grad_std_dct = result_manager.log_parameters(grad_stdev, 'post_train', 'sim',metric='grad_std')
    #     grad_sbm_dct = result_manager.log_parameters(grad_sigma_by_mu, 'post_train', 'sim', metric='grad_sigma_by_mu')

    #     result_manager.log_general_metric(FedstdevOptimizer.get_dict_avg(omegas, param_dims), f'omegas', 'sim', 'post_train')

    #     # Logging code ends here

    #     eval_result = trainer.eval()
    #     result_manager.log_general_result(eval_result, 'post_eval', 'sim', 'central_eval')

        # if curr_round % sim_cfg.checkpoint_every == 0:
        #     save_checkpoints()
        
        # This is weird, needs some rearch
        result_manager.flush_and_update_round(curr_round)

        loop_end = time.time() - loop_start
        logger.info(f'------------ Round {curr_round} completed in time: {loop_end} ------------')
    final_result = result_manager.finalize()
    return final_result



def run_flower_standalone_simulation(cfg: Config,
                                     client_datasets: fed_t.ClientDatasets_t,
                                     server_dataset: Subset,
                                     model: Module):
    pass
    
def run_standalone_simulation(cfg: Config,
                              client_datasets: fed_t.ClientDatasets_t,
                              server_dataset: Subset,
                              model: Module):
    
    central_model = deepcopy(model)
    clients: dict[str, BaseFlowerClient] = defaultdict()
    # Clients get the splits of the train set with an inbuilt test set
    all_client_ids = generate_client_ids(cfg.simulator.num_clients)

    make_checkpoint_dirs(has_server=False, client_ids=all_client_ids)
    test_loader = DataLoader(dataset=server_dataset,
                            batch_size=cfg.train_cfg.eval_batch_size, shuffle=False)
    # client_datasets = get_client_datasets(cfg.dataset.split_conf, train_set, test_set, match_train_distribution=False)

    result_manager = ResultManager(cfg.simulator, logger=logger)
    metric_manager  = MetricManager(cfg.train_cfg.metric_cfg, 0, actor='simulator')
    base_client_cfg = ClientSchema(
        _target_="feduciary.client.baseflowerclient.BaseFlowerClient",
        _partial_=True,
        cfg=cfg.client.cfg,
        train_cfg=cfg.train_cfg
                        )
    clients = create_clients(all_client_ids, client_datasets, model, base_client_cfg)

    central_train_cfg = instantiate(cfg.train_cfg)

    # exit(0)
    
    client_params = {cid: client.model.state_dict() for cid, client in clients.items()}

    _round = 0
    for curr_round in range(_round, cfg.simulator.num_rounds):
        logger.info(f'-------- Round: {curr_round} --------\n')
        train_result = {}
        eval_result = {}

        for cid, client in clients.items():
            logger.info(f'-------- Client: {cid} --------\n')
            client._round = curr_round
            client_train_ins = fed_t.ClientIns(params=client_params[cid],
                                         metadata={},
                                         request=fed_t.RequestType.TRAIN,
                                         _round=curr_round)
            outcome = client.download(client_train_ins)
            client_train_res = client.upload(fed_t.RequestType.TRAIN)
            client_params[cid] = client_train_res.params
            train_result[cid] = client_train_res.result

            eval_ins = fed_t.ClientIns(params=client_params[cid],
                                         metadata={},
                                         request=fed_t.RequestType.EVAL,
                                         _round=curr_round)
            
            outcome = client.download(eval_ins)
            eval_result[cid] = client.upload(fed_t.RequestType.EVAL).result

        aggregate_params = naive_aggregator(client_params)
        central_model.load_state_dict(aggregate_params)
        central_eval = simple_evaluator(central_model, test_loader, central_train_cfg, metric_manager, curr_round)

        result_manager.log_general_result(central_eval, phase='post_train', actor='sim', event='central_eval')
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
   
        logger.info(f'[SIM MODE] : {self.sim_cfg.mode}')
        logger.info(f'[SERVER] : {self.cfg.server._target_.split(".")[-1]}')
        logger.info(f'[STRATEGY] : {self.cfg.strategy._target_.split(".")[-1]}')
        logger.info(f'[CLIENT] : {self.cfg.client._target_.split(".")[-1]}')

        logger.info(f'[NUM ROUNDS] : {self.sim_cfg.num_rounds}')

        # self.num_clients = cfg.simulator.num_clients
        # self.all_client_ids = generate_client_ids(cfg.simulator.num_clients)

        set_seed(cfg.simulator.seed)
        # Remove calls like thes to make things more testable

        self.client_sets, self.test_set, self.model_instance = init_dataset_and_model(cfg=cfg)


        # NOTE: cfg object conversion to asdict breaks when init fields are not set
        if self.sim_cfg.use_wandb:
            wandb.init(project='fed_ml', job_type=cfg.mode,
                        config=asdict(cfg), resume=True, notes=cfg.desc)

        # self.metric_manager = MetricManager(cfg.client.cfg.metric_cfg, self._round, actor='simulator')

        # self.result_manager = ResultManager(cfg.simulator, logger=logger)

        # Till here can be factorized

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
                run_federated_simulation(self.cfg,
                                         self.client_sets,
                                         self.test_set,
                                         self.model_instance)
            case 'standalone':
                run_standalone_simulation(self.cfg,
                                         self.client_sets,
                                         self.test_set,
                                         self.model_instance)            
            case 'centralized':
                run_centralized_simulation(self.cfg,
                                         self.client_sets,
                                         self.test_set,
                                         self.model_instance)
            case 'flower':
                run_flower_simulation(self.cfg,
                                    self.client_sets,
                                    self.test_set,
                                    self.model_instance)
            case _:
                raise AssertionError(f'Mode: {self.sim_cfg.mode} is not implemented')

    
    def finalize(self):
        # if self.server:
        #     self.server.finalize()
        total_time= time.time() - self.start_time
        logger.info(f'Total runtime: {total_time} seconds')
        logger.info(f'Per loop runtime: {total_time/self.sim_cfg.num_rounds} seconds')

        logger.info('------------ Simulation completed ------------')

        # FIXME: Post process is broken
        # post_process(self.cfg, final_result, total_time=total_time)

        # del self.clients
        # del self.server
        logger.info('Closing Feduciary Simulator')
