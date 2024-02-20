import torch
import numpy as np
import os
import time
from itertools import combinations
from copy import deepcopy
from dataclasses import asdict, dataclass
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import wandb
import logging
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, RandomSampler
from torch import Generator, Tensor
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from icecream import install, ic

from feduciary.results.resultmanager import ResultManager
from feduciary.metrics.metricmanager import MetricManager
from feduciary.config import Config, register_configs, TrainConfig
from feduciary.simulator import set_seed
from feduciary.datasets import torchvisionparser
from feduciary.models.model import init_model
from feduciary.client.abcclient import simple_evaluator, simple_trainer
from feduciary.data import load_vision_dataset, load_raw_dataset
from feduciary.split import get_client_datasets, NoisySubset, LabelFlippedSubset
import feduciary.common.typing as fed_t

DEBUG = True
TRAIN_SHUFFLE = False
NUM_COPIES = 5
SEEDS = [15, 25, 35, 45, 55]
NUM_ELEMS = 40
logger = logging.getLogger(__name__)
install()
ic.configureOutput(includeContext=True)

def stratified_split(dataset: Dataset, split_pct: float) -> tuple[Subset, Subset]:
    '''Stratified split of the dataset'''
    pass

def n_model_copies(model, num_copies):
    return [deepcopy(model) for n in num_copies]

def create_shuffled_loaders(dataset: Subset, train_cfg: TrainConfig ) -> list[DataLoader]:
    loader_list = []
    for seed in SEEDS:
        gen = Generator()
        gen.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=gen)
        loader_list.append(DataLoader(dataset=dataset, sampler=sampler, batch_size=train_cfg.batch_size))
    return loader_list
    

def get_dict_avg(param_dict: dict, wts: dict) -> dict:
    # Helper function to compute average of the last layer of the dictionary
    wts_list = list(wts.values())

    avg = np.mean(list(param_dict.values()))
    wtd_avg = np.average(list(param_dict.values()), weights=wts_list)

    return {'avg': avg, 'wtd_avg': wtd_avg}

def compute_sigma_by_mu(sigma: Tensor, mu: Tensor) -> Tensor:
    mask = (mu.data!=0)
    sbm = sigma.clone()

    sbm[mask] = sigma[mask]/ mu[mask].abs()
    return sbm

def compute_sigma_by_mu_full(sigmas: fed_t.ActorDeltas_t, mus: fed_t.ActorDeltas_t) -> fed_t.ActorDeltas_t:
        '''Computes sigma/mu for each parameter'''
        # assert(sigmas.keys() == mus.keys())
        sigma_by_mu = {}
        for (key, sigma), mu in zip (sigmas.items(), mus.values()):
            sigma_by_mu[key] = compute_sigma_by_mu(sigma, mu)
        return sigma_by_mu
    

def aggregate_model(model_list: list[nn.Module], grad_list: list, resman: ResultManager) -> nn.Module:
    model = deepcopy(model_list[0])

    param_dims = {name : np.prod(param.data.shape) for name, param in model.named_parameters()}
    mean_params = {name: torch.empty_like(param.data) for name, param in model.named_parameters()}
    std_params = deepcopy(mean_params)
    mean_grads = deepcopy(mean_params)
    std_grads = deepcopy(mean_params)

    grad_cos_sim = {}
    grad_cos_mean = {}
    grad_cos_stds = {}

    param_norms = {}
    grad_norms = {}
    grad_norm_std = {}

    # grad_sbm_norms = {}
    # grad_sbm

    # Layer wise aggregation
    for name, param in model.named_parameters():
        tmp_param_list = []
        tmp_grad_list = []
        flattened_grads_list = []
        flattened_params_list = []
        grad_cos = []

        for model, grad in zip(model_list, grad_list):
            individual_param = model.get_parameter(name)
            tmp_param_list.append(individual_param.data)
            flattened_params_list.append(parameters_to_vector(individual_param))
            # tmp_grad_list.append(individual_param.grad)
            tmp_grad_list.append(grad[name])
            flattened_grads_list.append(parameters_to_vector(grad[name]))

        # parameter average
        stacked = torch.stack(tmp_param_list)
        # mean_ = torch.mean(stacked, dim=0)
        std_, mean_ = torch.std_mean(stacked, dim=0)
        # ic(std_.shape, mean_.shape)
        mean_params[name] = mean_
        std_params[name] = std_

        # gradient average and std dev
        stacked_grad = torch.stack(tmp_grad_list)

        flattened_grads = torch.stack(flattened_grads_list)
        flattened_params = torch.stack(flattened_params_list)
        
        grad_norm_ = torch.norm(flattened_grads, dim=1)
        grad_norm_std_, grad_norm_mean_ = torch.std_mean(grad_norm_, dim=0)
        grad_norms[name] = grad_norm_mean_.item()
        grad_norm_std[name] = grad_norm_std_.item()

        param_norm_ = torch.norm(flattened_params, dim=1)
        param_norms[name] = torch.mean(param_norm_).item()

        
    
        for g1, g2 in combinations(flattened_grads_list, 2):
            grad_cos.append(torch.cosine_similarity(g1, g2, dim=0).item())
        
        grad_cos_sim[name] = grad_cos
        # grad_cos_stacked = torch.stack(grad_cos)
        mean_grad_cos, std_grad_cos = np.mean(grad_cos), np.std(grad_cos)
        grad_cos_mean[name] = mean_grad_cos
        grad_cos_stds[name] = std_grad_cos

        # std_grad_, mean_grad_ = torch.std_mean(stacked_grad.abs(), dim=0)
        std_grad_, mean_grad_ = torch.std_mean(stacked_grad, dim=0)
        mean_grads[name] = mean_grad_
        std_grads[name] = std_grad_

        param.data.copy_(mean_.data)
        param.grad = mean_grad_.data


    resman.log_parameters(mean_params, phase='post_train', actor='sim', metric='param_mean')
    resman.log_parameters(std_params, phase='post_train', actor='sim', metric='param_std')
    resman.log_parameters(mean_grads, phase='post_train', actor='sim', metric='grad_mean')
    resman.log_parameters(std_grads, phase='post_train', actor='sim', metric='grad_std')

    param_sbm = compute_sigma_by_mu_full(std_params, mean_params)
    resman.log_parameters(param_sbm, phase='post_train', actor='sim', metric='param_sigma_by_mu')

    grad_sbm = compute_sigma_by_mu_full(std_grads, mean_grads)
    resman.log_parameters(grad_sbm, phase='post_train', actor='sim', metric='grad_sigma_by_mu')

    grad_sbm_norms = lump_tensors_norms(grad_sbm)
    param_sbm_norms = lump_tensors_norms(param_sbm)

    param_norms = get_dict_avg(param_norms, param_dims)
    grad_norms = get_dict_avg(grad_norms, param_dims)
    grad_norm_std = get_dict_avg(grad_norm_std, param_dims)
    grad_cos_mean = get_dict_avg(grad_cos_mean, param_dims)
    grad_cos_stds = get_dict_avg(grad_cos_stds, param_dims)

    # out_dict_params = {'param_mean': mean_params, 'param_std': std_params,
                    #    'mean': mean_grads, 'grad_std': std_grads} 
    
    resman.log_general_metric(param_norms, phase='post_train', actor='sim', metric_name='param_norms')
    resman.log_general_metric(grad_norms, phase='post_train', actor='sim', metric_name='grad_norms')
    resman.log_general_metric(grad_norm_std, phase='post_train', actor='sim', metric_name='grad_norm_std')
    resman.log_general_metric(grad_cos_mean, phase='post_train', actor='sim', metric_name='grad_cos_mean')
    resman.log_general_metric(grad_cos_stds, phase='post_train', actor='sim', metric_name='grad_cos_stds')

    
    # out_dict_metrics = {'param_norms': param_norms, 
    #             'grad_norms_mean': grad_norms, 'grad_norms_std': grad_norm_std,
    #             'grad_cos_mean': grad_cos_mean, 'grad_cos_std': grad_cos_stds}
    

    return model

def compute_model_wide_norm_and_cosine_similarity(param_list: list[fed_t.ActorDeltas_t]) -> tuple[dict[str, float], dict[str, float]]:
    # Compute the model wide norm and cosine similarity
    # Compute the model wide norm and cosine similarity
    norms = {}
    cosines = {}
    vecs = []
    for sd, param in zip(SEEDS, param_list):
        vec = parameters_to_vector(param.values())
        norms[sd] = torch.norm(vec).item()
        vecs.append(vec)
    
    norms['mean'] = np.mean(list(norms.values()))
    norms['std'] = np.std(list(norms.values()))

    for (s1, s2), (v1, v2) in zip(combinations(SEEDS, 2), combinations(vecs, 2)):
        cosines[f'seed_{s1}_{s2}'] = torch.cosine_similarity(v1, v2, dim=0).item()
    cosines['mean'] = np.mean(list(cosines.values()))
    cosines['std'] = np.std(list(cosines.values()))
    
    return norms, cosines

def aggregate_results(seed_results: list[fed_t.Result]) -> fed_t.Result:
    # Results aggregations
    _round = seed_results[0]._round
    sample_res = fed_t.Result()
    avg_metrics = {}
    for res in seed_results:
        for metric, val in res.metrics.items():
            avg_metrics[metric] = avg_metrics.get(metric, 0) + val
        sample_res = res

    for metric, val in avg_metrics.items():
        avg_metrics[metric] = val / len(seed_results)
    
    return fed_t.Result(metrics=avg_metrics, size=sample_res.size, metadata=sample_res.metadata,event=sample_res.event, phase=sample_res.phase, _round=_round, actor='sim')

def compute_model_wide_norm_and_cosine_similarity_no_seed(param_list: list[fed_t.ActorDeltas_t]) -> tuple[dict[str, float], dict[str, float]]:
    # Compute the model wide norm and cosine similarity
    # Compute the model wide norm and cosine similarity
    norms = {}
    cosines = {}
    cos_list = []
    norms_list = []

    vecs = []
    for param in  param_list:
        vec = parameters_to_vector(param.values())
        norms_list[torch.norm(vec).item()]
        vecs.append(vec)
    
    norms['mean'] = np.mean(norms_list)
    norms['std'] = np.std(norms_list)

    for (v1, v2) in combinations(vecs, 2):
        cos_list.append(torch.cosine_similarity(v1, v2, dim=0).item())

    cosines['mean'] = np.mean(cos_list)
    cosines['std'] = np.std(cos_list)
    
    return norms, cosines

def aggregate_gradients(grads_list: list[fed_t.ActorDeltas_t], param_dims:dict, resman: ResultManager) -> None:
    '''Aggregates the gradients from the different models'''
    # Aggregates the gradients from the different models

    mean_grads = deepcopy(grads_list[0])
    std_grads = deepcopy(mean_grads)

    grad_cos_sim = {}
    grad_cos_mean = {}
    grad_cos_stds = {}

    grad_norms = {}
    grad_norm_std = {}

    # grad_sbm_norms = {}
    # grad_sbm

    # Layer wise aggregation
    for name, param_dim in param_dims.items():
        tmp_grad_list = []
        flattened_grads_list = []
        grad_cos = []

        for grad in grads_list:
            # tmp_grad_list.append(individual_param.grad)
            tmp_grad_list.append(grad[name])
            flattened_grads_list.append(parameters_to_vector(grad[name]))

        # gradient average and std dev
        stacked_grad = torch.stack(tmp_grad_list)

        flattened_grads = torch.stack(flattened_grads_list)
        
        grad_norm_ = torch.norm(flattened_grads, dim=1)
        grad_norm_std_, grad_norm_mean_ = torch.std_mean(grad_norm_, dim=0)
        grad_norms[name] = grad_norm_mean_.item()
        grad_norm_std[name] = grad_norm_std_.item()

        for g1, g2 in combinations(flattened_grads_list, 2):
            grad_cos.append(torch.cosine_similarity(g1, g2, dim=0).item())
        ic(len(grad_cos))
        grad_cos_sim[name] = grad_cos
        # grad_cos_stacked = torch.stack(grad_cos)
        mean_grad_cos, std_grad_cos = np.mean(grad_cos), np.std(grad_cos)
        grad_cos_mean[name] = mean_grad_cos
        grad_cos_stds[name] = std_grad_cos

        # std_grad_, mean_grad_ = torch.std_mean(stacked_grad.abs(), dim=0)
        std_grad_, mean_grad_ = torch.std_mean(stacked_grad, dim=0)
        mean_grads[name] = mean_grad_
        std_grads[name] = std_grad_

    resman.log_parameters(mean_grads, phase='post_train', actor='sim', metric='grad_mean')
    resman.log_parameters(std_grads, phase='post_train', actor='sim', metric='grad_std')

    grad_sbm = compute_sigma_by_mu_full(std_grads, mean_grads)
    resman.log_parameters(grad_sbm, phase='post_train', actor='sim', metric='grad_sigma_by_mu')

    grad_sbm_norms = lump_tensors_norms(grad_sbm)
    resman.log_general_metric(grad_sbm_norms, phase='post_train', actor='sim', metric_name='grad_sbm_norms')


    grad_norms = get_dict_avg(grad_norms, param_dims)
    grad_norm_std = get_dict_avg(grad_norm_std, param_dims)
    grad_cos_mean = get_dict_avg(grad_cos_mean, param_dims)
    grad_cos_stds = get_dict_avg(grad_cos_stds, param_dims)

    resman.log_general_metric(grad_norms, phase='post_train', actor='sim', metric_name='grad_norms')
    resman.log_general_metric(grad_norm_std, phase='post_train', actor='sim', metric_name='grad_norm_std')
    resman.log_general_metric(grad_cos_mean, phase='post_train', actor='sim', metric_name='grad_cos_mean')
    resman.log_general_metric(grad_cos_stds, phase='post_train', actor='sim', metric_name='grad_cos_stds')



def trainer_mono(model: nn.Module, loader: DataLoader, train_cfg: TrainConfig, mm: MetricManager, r: int, resman: ResultManager) -> tuple[fed_t.Result, nn.Module]:
    mm._round = r
    model.train()
    model.to(train_cfg.device)
    optimizer: Optimizer =  train_cfg.optimizer(model.parameters())
    # empty_grads = {p_key: torch.empty_like
    #                (param.data, device=train_cfg.device) for p_key, param in model.named_parameters()}
    
    grads = []
    n_iters = len(loader.dataset)//train_cfg.batch_size 
    record_every = n_iters//NUM_ELEMS
    ic(record_every)
    # ic(len(loader.dataset))
    # ic(n_iters)

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(train_cfg.device), targets.to(train_cfg.device)

        optimizer.zero_grad(set_to_none=True)

        outputs: Tensor = model(inputs)
        loss: Tensor = train_cfg.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # for p_key, param in model.named_parameters():
        #     add_grad = 0 if param.grad is None else param.grad
        
        # grads.append({k: param.grad.to('cpu') for k, param in model.named_parameters()})
        if i%record_every== 0:
            grads.append({k: param.grad for k, param in model.named_parameters()})

        mm.track(loss.item(), outputs, targets)
    else:
        # ic(i)
        res = mm.aggregate(len(loader.dataset), 0) # type: ignore
        mm.flush()
    
    # model.to('cpu')

    ic(len(grads))
    grad_model_norm, grad_model_cos = compute_model_wide_norm_and_cosine_similarity_no_seed(grads)

    resman.log_general_metric(grad_model_norm, phase='post_train', actor='sim', metric_name='grad_model_norm')
  
    resman.log_general_metric(grad_model_cos, phase='post_train', actor='sim', metric_name='grad_model_cos')
    
    param_dims = {name : np.prod(param.data.shape) for name, param in model.named_parameters()}

    aggregate_gradients(grads, param_dims, resman)

    return res, model

def trainer(models: list[nn.Module],
             loaders: list[DataLoader],
             train_cfg: TrainConfig, mm: MetricManager, r: int,
             resman: ResultManager) -> tuple[fed_t.Result, nn.Module]:
    
    mm._round = r
    model_0 = deepcopy(models[0])
    empty_grads = {p_key: torch.zeros_like(param.data, device=train_cfg.device) for p_key, param in model_0.named_parameters()}
    grads = [deepcopy(empty_grads) for n in range(NUM_COPIES)]

    optim_part = train_cfg.optimizer
    # train_cfg.criterion = instantiate(train_cfg.criterion)

    results = []
    losses = []

    grad_mags = []
    grad_cos = []

    for i, (model, loader )in enumerate(zip(models, loaders)):
        model.train()
        model.to(train_cfg.device)
        optimizer: Optimizer =  optim_part(model.parameters())
        # for epoch in range(train_cfg.epochs):
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(train_cfg.device), targets.to(train_cfg.device)

            optimizer.zero_grad(set_to_none=True)

            outputs: Tensor = model(inputs)
            loss: Tensor = train_cfg.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            for p_key, param in model.named_parameters():
                add_grad = 0 if param.grad is None else param.grad
                grads[i][p_key] = grads[i].get(p_key) + add_grad # type: ignore

            mm.track(loss.item(), outputs, targets)
        else:
            res = mm.aggregate(len(loader.dataset), 0) # type: ignore
            results.append(res)
            losses.append(res.metrics['loss'])
            mm.flush()
    
    grad_model_norm, grad_model_cos = compute_model_wide_norm_and_cosine_similarity(grads)
    param_model_norm, param_model_cos = compute_model_wide_norm_and_cosine_similarity([model.state_dict() for model in models])

    resman.log_general_metric(grad_model_norm, phase='post_train', actor='sim', metric_name='grad_model_norm')
  
    resman.log_general_metric(grad_model_cos, phase='post_train', actor='sim', metric_name='grad_model_cos')

    resman.log_general_metric(param_model_norm, phase='post_train', actor='sim', metric_name='param_model_norm')

    resman.log_general_metric(param_model_cos, phase='post_train', actor='sim', metric_name='param_model_cos')

    out_model= aggregate_model(models, grads, resman)

    # set the model to the average model
    for model in models:
        model.load_state_dict(out_model.state_dict())

    out_result = aggregate_results(results)

    return out_result, out_model

def lump_abs_mean_tensors(in_dict: dict[str, Tensor]) -> dict[str, float]:
    '''Lumps all the tensors in the dictionary into a single value per key'''
    return {key: val.abs().mean().item() for key, val in in_dict.items()}

def lump_tensors_norms(in_dict: dict[str, Tensor]) -> dict[str, float]:
    '''Lumps all the tensors in the dictionary into a single value per key'''
    return {key: val.norm().item() for key, val in in_dict.items()}



def run_sandbox(cfg: Config):
    sim_cfg = cfg.simulator
    train_cfg = cfg.train_cfg
    set_seed(cfg.simulator.seed)

    if sim_cfg.use_wandb:
        wandb.init(project='fed_ml', job_type=cfg.mode,
                        config=asdict(cfg), resume=True, notes=cfg.desc)
    
    train_set, test_set, dataset_model_spec  = load_raw_dataset(cfg.dataset)
    cfg.model.model_spec.in_channels = dataset_model_spec.in_channels
    cfg.model.model_spec.num_classes = dataset_model_spec.num_classes

    model = init_model(cfg.model)

    res = ResultManager(sim_cfg, logger)
    mm = MetricManager(train_cfg.metric_cfg, 0, 'sim')
    split_cfg = cfg.dataset.split_conf

    match split_cfg.split_type:
        case 'n_label_flipped_clients':
            train_set = LabelFlippedSubset(train_set, flip_pct=split_cfg.noise.flip_percent) #type: ignore
        case 'n_noisy_clients':
            train_set = NoisySubset(train_set, split_cfg.noise.mu, split_cfg.noise.sigma) #type: ignore
        case 'iid':
            pass
        case _:
            logger.error('Unknown datasest type')
            raise ValueError
    
    model_list = [deepcopy(model) for n in range(NUM_COPIES)]

    loader_list = create_shuffled_loaders(train_set, train_cfg)
    train_loader = DataLoader(train_set, train_cfg.batch_size, shuffle=TRAIN_SHUFFLE)
    test_loader = DataLoader(test_set, train_cfg.eval_batch_size, shuffle=False)

    train_cfg = instantiate(train_cfg)

    for r in range(sim_cfg.num_rounds):
        logger.info(f'-------- Round: {r} --------\n')
        loop_start = time.time()


        # train_res, model = trainer(model_list, loader_list, train_cfg, mm, r, res)

        train_res, model = trainer_mono(model, train_loader, train_cfg, mm, r, res) 

        eval_res = simple_evaluator(model, test_loader, train_cfg, mm, r)

        # Log the results
        # res.log_general_metric(grads, phase='post_train', actor='sim', event='grad_mean')
        res.log_general_result(train_res, phase='post_train', actor='sim', event='central_train')

        res.log_general_result(eval_res, phase='post_train', actor='sim', event='central_eval')
        res.flush_and_update_round(r)
        loop_end = time.time() - loop_start
        logger.info(f'------------ Round {r} completed in time: {loop_end} ------------\n')


@hydra.main(version_base=None, config_path="../conf", config_name="sandbox_config")
def run_feduciary_sandbox(cfg: Config):
    cfg_obj: Config = OmegaConf.to_object(cfg) #type:ignore
    logger.debug((OmegaConf.to_yaml(cfg)))

    run_sandbox(cfg_obj)


if __name__ == '__main__':
    register_configs()
    run_feduciary_sandbox()