import typing as t
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import json
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from torch.utils.data import Dataset, RandomSampler, DataLoader, Subset
from torch import Generator, tensor, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from .baseflowerclient import  BaseFlowerClient, MetricManager, ClientConfig
from feduciary.strategy.fedgradstdstrategy import FedgradstdStrategy, FedgradIns
from feduciary.config import FedgradstdClientConfig
from feduciary.common.utils import get_time
from feduciary.common import typing as fed_t

logger = logging.getLogger(__name__)



def get_to_nearest_log2(num, quotient):
    max_batch = num//quotient
    exp = int(np.log2(max_batch))
    new_batch = 2**exp
    while num//new_batch != quotient:
        residue = max_batch - new_batch
        new_batch = new_batch + 2**int(np.log2(residue))
    return new_batch


@dataclass
class ClientInProtocol(t.Protocol):
    in_params: dict

# @dataclass
# class ClientOuts:
#     client_params: fed_t.ActorParams_t
#     client_grad_mus: fed_t.ActorDeltas_t
#     client_grad_stds: fed_t.ActorDeltas_t
    

class FedgradstdClient(BaseFlowerClient):
    # TODO: Fix the argument ordering structure
    def __init__(self, cfg: FedgradstdClientConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = deepcopy(cfg)
        # self._root_dir = f'{self.cfg.metric_cfg.cwd}/temp_json'
        # os.makedirs(self._root_dir, exist_ok=True)

        # Keep n iters consistent with the iid split
        if cfg.n_iters -1 != len(self.training_set)//self.train_cfg.batch_size:
            logger.debug(f'NITERS_BEFORE: {len(self.training_set)//self.train_cfg.batch_size +1}')
            self.train_cfg.batch_size = get_to_nearest_log2(len(self.training_set), cfg.n_iters -1)
            logger.debug(f'NITERS_AFTER: {len(self.training_set)//self.train_cfg.batch_size +1}')

        logger.debug(f'[BATCH SIZES:] CID: {self._cid}, batch size: {self.train_cfg.batch_size}')

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
        # self._model_map: dict[int, Module] = {}
        # Make m copies of the model for independent train iterations
        self._model_map: dict[int, Module] = {seed: deepcopy(self._model) for seed in cfg.seeds}


        self._optimizer_map: dict[int, Optimizer] = {}
        for seed, model in self._model_map.items():
            self._optimizer_map[seed] = self.optim_partial(model.parameters())

        # self._param_std :fed_t.ActorParams_t  = self._model.state_dict()
        self._grad_mu : fed_t.ActorDeltas_t = {p_key: torch.empty_like(param.data) for p_key, param in self._model.named_parameters()}

        # self._empty_grads = deepcopy(self._grad_mu)
        # self._cum_gradients_map = {seed: deepcopy(self._empty_grads) for seed in self._model_map.keys()}

        self._grad_std :dict[str, Tensor] = deepcopy(self._grad_mu)



    def _create_shuffled_loaders(self, dataset: Subset, seeds:list[int]) -> dict[int, DataLoader]:
        loader_dict = {}
        self._generator = {}
        index_list = []
        # with get_time():
        for seed in seeds:
            gen = Generator()
            gen.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
            self._generator[seed] = gen
            loader_dict[seed] = DataLoader(dataset=dataset, sampler=sampler, batch_size=self.train_cfg.batch_size)

            index_list.append(np.array(loader_dict[seed].dataset.indices))
        
        # for i in range(1, len(index_list)):
        #     i1 = np.sort(index_list[i-1])
        #     i2 = np.sort(index_list[i])
        #     # ic(i1[:10], i2[:10] )
        #     if not np.array_equal(i1, i2):
        #         raise ValueError('Indices are not equal')
        return loader_dict
    
    def unpack_train_input(self, client_ins: fed_t.ClientIns) -> ClientInProtocol:
        specific_ins = FedgradstdStrategy.client_receive_strategy(client_ins)
        return specific_ins
    
    def pack_train_result(self, result: fed_t.Result) -> fed_t.ClientResult:
        self._model.to('cpu')
        client_outs = FedgradIns(
                    client_params=self._model.state_dict(),
                    client_grad_mus=self._get_gradients_average(),
                    client_grad_stds=self._get_gradients_std_dev()
                )
        specific_res = FedgradstdStrategy.client_send_strategy(client_outs, result=result)
        return specific_res
 
    
    def _get_gradients_std_dev(self)->dict[str, Tensor]:
        return deepcopy(self._grad_std)
    
    def _get_gradients_average(self)->dict[str, Tensor]:
        return deepcopy(self._grad_mu)

    def _compute_average_model_and_std(self, model_map: dict[int, Module]):        

        for name, param in self._model.named_parameters():
            tmp_param_list = []
            tmp_grad_list = []

            for seed, model in model_map.items():
                individual_param = model.get_parameter(name)
                tmp_param_list.append(individual_param.data)
                # tmp_grad_list.append(individual_param.grad)
                tmp_grad_list.append(self._cum_gradients_map[seed][name])

            # parameter average
            stacked = torch.stack(tmp_param_list)
            mean_ = torch.mean(stacked, dim=0)
            # std_, mean_ = torch.std_mean(stacked, dim=0)

            # gradient average and std dev
            stacked_grad = torch.stack(tmp_grad_list)
            if self.cfg.abs_before_mean:
                std_grad_, mean_grad_ = torch.std_mean(stacked_grad.abs(), dim=0)
            else:
                std_grad_, mean_grad_ = torch.std_mean(stacked_grad, dim=0)

            param.data.copy_(mean_.data)

            param.grad = mean_grad_.data

            # STATEFUL FUNCTIONS MAY NOT WORK WITH MULTITHREADING
            # self._param_std[name].data = std_.to('cpu')
            self._grad_mu[name] = mean_grad_.to('cpu')
            self._grad_std[name] = std_grad_.to('cpu')


    def aggregate_seed_results(self, seed_results: dict[int, fed_t.Result]) -> fed_t.Result:
        # TODO: Consider merging this with the above function
        sample_res = fed_t.Result()
        avg_metrics = {}
        for seed, res in seed_results.items():
            for metric, val in res.metrics.items():
                avg_metrics[metric] = avg_metrics.get(metric, 0) + val
            sample_res = res

        for metric, val in avg_metrics.items():
            avg_metrics[metric] = val / len(seed_results)
        
        return fed_t.Result(metrics=avg_metrics, size=sample_res.size, metadata=sample_res.metadata,event=sample_res.event, phase=sample_res.phase, _round=self._round, actor=self._cid)
    
    def train(self, train_ins: ClientInProtocol) -> fed_t.Result:
        # MAYBE THIS PART IS REDUNDANT
        self._model.load_state_dict(train_ins.in_params)
        self._optimizer = self.optim_partial(self._model.parameters())

        # NOTE: It is important to reseed the generators here to ensure the tests pass across flower and non flower runs.
        self.train_loader_map = self._create_shuffled_loaders(self.training_set, self.cfg.seeds)
        for seed, model in self._model_map.items():
            model.load_state_dict(train_ins.in_params)
            self._optimizer_map[seed] = self.optim_partial(model.parameters())
            # self.metric_mngr.json_dump(self.train_loader_map[seed].dataset.indices.tolist(), 'indices', 'train', f'seed_{seed}')
        # Run an round on the client
        empty_grads = {p_key: torch.empty_like(param.data, device=self.train_cfg.device) for p_key, param in self._model.named_parameters()}

        self._cum_gradients_map = {seed: deepcopy(empty_grads) for seed in self._model_map.keys()}

        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.train_cfg.device)

        resume_epoch = 0
        # out_result= fed_t.Result()
        out_result_dict: dict[int, fed_t.Result] = {}

        for seed, model in self._model_map.items():  
            # logger.info(f'SEED: {seed}, STATE: {self._generator[seed].get_state()}')
            # set optimizer parameters
            # optimizer: Optimizer = self.optim_partial(model.parameters())
            optimizer: Optimizer = self._optimizer_map[seed]

            model.train()
            model.to(self.train_cfg.device)
            # iterate over epochs and then on the batches
            for epoch in range(resume_epoch, resume_epoch + self.train_cfg.epochs):
                for i, (inputs, targets) in enumerate(self.train_loader_map[seed]):
                    # with open(f'{self._root_dir}/{self.cfg.metric_cfg.file_prefix}_targets_{self._round}_{seed}_{i}.json', 'w') as f:
                    #     json.dump(targets.tolist(), f)
                    # logger.debug(f'CLIENT {self.id} SEED: {seed}, EPOCH: {epoch}, BATCH: {i}')

                    inputs, targets = inputs.to(self.train_cfg.device), targets.to(self.train_cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs: Tensor = model(inputs)
                    loss: Tensor = self.criterion(outputs, targets) #type: ignore
                    loss.backward()
                    optimizer.step()

                    for p_key, param in model.named_parameters():
                        add_grad = 0 if param.grad is None else param.grad
                        self._cum_gradients_map[seed][p_key] = self._cum_gradients_map[seed].get(p_key, 0) + add_grad # type: ignore
      
                    self.metric_mngr.track(loss.item(), outputs, targets)
                else:
                    out_result_dict[seed] = self.metric_mngr.aggregate(len(self.training_set), epoch)
                    self.metric_mngr.flush()

                self._epoch = epoch

        self._compute_average_model_and_std(self._model_map)
        out_result = self.aggregate_seed_results(out_result_dict)
 
        self._start_epoch = self._epoch + 1

        logger.info(f'CLIENT {self.id} Completed update')

        for seed, out_result in out_result_dict.items():
            self.metric_mngr.log_general_metric(out_result.metrics['loss'], 'train_loss', f'seed_{seed}', 'pre_avg')
            self.metric_mngr.log_general_metric(out_result.metrics['acc1'], 'train_acc', f'seed_{seed}', 'pre_avg')
        
        # self.metric_mngr.json_dump(self._get_parameter_std_dev(), 'param_std', 'train', 'pre_avg')
        # self.metric_mngr.json_dump(self._parameters, 'grad_std', 'train', 'pre_avg')
        
        return out_result

    def load_checkpoint(self, ckpt_path):
        super().load_checkpoint(ckpt_path)
        self._model_map = {seed: deepcopy(self._model) for seed in self.cfg.seeds}

        self._optimizer_map = {seed: deepcopy(self._optimizer) for seed in self.cfg.seeds}


 