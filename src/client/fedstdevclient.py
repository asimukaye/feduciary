import typing as t
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import json
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset, RandomSampler, DataLoader, Subset
from torch import Generator, tensor, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from .baseflowerclient import  BaseFlowerClient, MetricManager, ClientConfig
from src.strategy.fedstdevstrategy import FedstdevStrategy
from src.config import FedstdevClientConfig, TrainConfig
from src.common.utils import get_time
import src.common.typing as fed_t

logger = logging.getLogger(__name__)
 # NOTE: Multithreading causes atleast 3x slowdown for 2 epoch case. DO not use until necessary
def train_one_model(model:Module, dataloader: DataLoader, seed: int, cfg: TrainConfig, optim_partial, criterion, mm: MetricManager)->t.Tuple[int, Module,fed_t.Result]:
    out_result = fed_t.Result()
    optimizer: Optimizer = optim_partial(model.parameters())
    model.train()
    model.to(cfg.device)
    # iterate over epochs and then on the batches
    for _epoch in range(cfg.epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

            model.zero_grad(set_to_none=True)

            outputs: Tensor = model(inputs)
            loss: Tensor = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # accumulate metrics
            mm.track(loss.item(), outputs, targets)
        else:
            out_result = mm.aggregate(len(
                list(dataloader.dataset)), _epoch)
            mm.flush()
    return (seed, model, out_result)

@dataclass
class ClientInProtocol(t.Protocol):
    server_params: dict

@dataclass
class ClientOuts:
    client_params: fed_t.ActorParams_t
    client_param_stds: fed_t.ActorParams_t
    

class FedstdevClient(BaseFlowerClient):
    # TODO: Fix the argument ordering structure
    def __init__(self, cfg: FedstdevClientConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = deepcopy(cfg)
        # self._root_dir = f'{self.cfg.metric_cfg.cwd}/temp_json'
        # os.makedirs(self._root_dir, exist_ok=True)

        self.train = self.train_single_thread

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
        # self._model_map: dict[int, Module] = {}
        # Make m copies of the model for independent train iterations
        self._model_map: dict[int, Module] = {seed: deepcopy(self._model) for seed in cfg.seeds}

        self._optimizer_map: dict[int, Optimizer] = {}
        for seed, model in self._model_map.items():
            self._optimizer_map[seed] = self.optim_partial(model.parameters())

        self._param_std :fed_t.ActorParams_t  = self._model.state_dict()
        self._grad_mu : fed_t.ActorDeltas_t = {p_key: torch.empty_like(param.data) for p_key, param in self._model.named_parameters()}

        # self._empty_grads = deepcopy(self._grad_mu)
        # self._cum_gradients_map = {seed: deepcopy(self._empty_grads) for seed in self._model_map.keys()}

        self._grad_std :dict[str, Tensor] = deepcopy(self._grad_mu)



    def _create_shuffled_loaders(self, dataset: Subset, seeds:list[int]) -> dict[int, DataLoader]:
        loader_dict = {}
        self._generator = {}
        # HACK: Hack to fix the batch size based on dataset size for imbalanced dataset
        # FIXME: IMPORTANT: Formalize the iteration to batch size correlation for imbalanced data experiments
        for seed in seeds:
            gen = Generator()
            gen.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
            self._generator[seed] = gen
            loader_dict[seed] = DataLoader(dataset=dataset, sampler=sampler, batch_size=self.train_cfg.batch_size)
            # for i, (inputs, targets) in enumerate(loader_dict[seed]):
            #     with open(f'{self._root_dir}/{self.cfg.metric_cfg.file_prefix}_targets_pre_{self._round}_{seed}_{i}.json', 'w') as f:
            #                 json.dump(targets.tolist(), f)



        return loader_dict
    
    def unpack_train_input(self, client_ins: fed_t.ClientIns) -> ClientInProtocol:
        specific_ins = FedstdevStrategy.client_receive_strategy(client_ins)
        return specific_ins
    
    def pack_train_result(self, result: fed_t.Result) -> fed_t.ClientResult1:
        self._model.to('cpu')
        client_outs = ClientOuts(
                    client_params=self._model.state_dict(),
                    client_param_stds=self._get_parameter_std_dev()
                )
        specific_res = FedstdevStrategy.client_send_strategy(client_outs, result=result)
        return specific_res
 

    def _get_parameter_std_dev(self)->dict[str, Parameter]:
        return deepcopy(self._param_std)
    
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


            stacked = torch.stack(tmp_param_list)

            std_, mean_ = torch.std_mean(stacked, dim=0)

            stacked_grad = torch.stack(tmp_grad_list)
            std_grad_, mean_grad_ = torch.std_mean(stacked_grad, dim=0)

            param.data.copy_(mean_.data)
            # with get_time():
            param.grad = mean_grad_.data

            self._param_std[name].data = std_.to('cpu')
            self._grad_mu[name] = mean_grad_.to('cpu')
            self._grad_std[name] = std_grad_.to('cpu')

        # Tie down the other models to the common model
        # with get_time():
        # new_state_dict = self._model.state_dict()
        # for seed, model in model_map.items():
        #     model.load_state_dict(new_state_dict)
            # self._optimizer_map[seed] = self.optim_partial(model.parameters())


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
    
    def train_single_thread(self, train_ins: ClientInProtocol) -> fed_t.Result:
        # MAYBE THIS PART IS REDUNDANT
        self._model.load_state_dict(train_ins.server_params)
        self._optimizer = self.optim_partial(self._model.parameters())

        # NOTE: It is important to reseed the generators here to ensure the tests pass across flower and non flower runs. Commenting for performance purposes
        # self.train_loader_map = self._create_shuffled_loaders(self.training_set, self.cfg.seeds)
        for seed, model in self._model_map.items():
            model.load_state_dict(train_ins.server_params)
            self._optimizer_map[seed] = self.optim_partial(model.parameters())

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

                    inputs, targets = inputs.to(self.train_cfg.device), targets.to(self.train_cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs: Tensor = model(inputs)
                    loss: Tensor = self.criterion(outputs, targets)
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
        # FIXME: OUt result is only for the last seed
        return out_result

    def load_checkpoint(self, ckpt_path):
        super().load_checkpoint(ckpt_path)
        self._model_map = {seed: deepcopy(self._model) for seed in self.cfg.seeds}

        self._optimizer_map = {seed: deepcopy(self._optimizer) for seed in self.cfg.seeds}

   
    def train_multi_thread(self):
        # Run an round on the client
    
        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.train_cfg.device)
        out_result = fed_t.Result()
        # for seed, model in self._model_map.items():   
        with ThreadPoolExecutor(max_workers=len(self._model_map)) as exec:
            futures = {exec.submit(train_one_model, model, self.train_loader_map[seed], seed, self.train_cfg, self.optim_partial, self.criterion, deepcopy(self.metric_mngr)) for seed, model in self._model_map.items()}
            for fut in as_completed(futures):
                seed, model, out_result = fut.result()
                self._model_map[seed] = model
        
        self._compute_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')
        return out_result
 