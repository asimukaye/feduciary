import typing as t
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset, RandomSampler, DataLoader, Subset
from torch import Generator, tensor, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from .baseflowerclient import  BaseFlowerClient, MetricManager, ClientConfig
from src.strategy.fedstdevstrategy import FedstdevStrategy
from src.config import FedstdevClientConfig
from src.common.utils import get_time
import src.common.typing as fed_t

logger = logging.getLogger(__name__)
 # NOTE: Multithreading causes atleast 3x slowdown for 2 epoch case. DO not use until necessary
def train_one_model(model:Module, dataloader: DataLoader, seed: int, cfg: ClientConfig, optim_partial, criterion, mm: MetricManager)->t.Tuple[int, Module,fed_t.Result]:
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
        # self.res_man: ResultManager = kwargs.get('res_man')

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
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

        self.train = self.train_single_thread


    def _create_shuffled_loaders(self, dataset: Subset, seeds:list[int]) -> dict[int, DataLoader]:
        loader_dict = {}
        # HACK: Hack to fix the batch size based on dataset size for imbalanced dataset
        # FIXME: IMPORTANT: Formalize the iteration to batch size correlation for imbalanced data experiments
        for seed in seeds:
            gen = Generator()
            gen.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
            loader_dict[seed] = DataLoader(dataset=dataset, sampler=sampler, batch_size=self.cfg.batch_size)

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
    
    # def download(self, client_ins: fed_t.ClientIns) -> fed_t.RequestOutcome:
    #     # Download initiates training. Is a blocking call without concurrency implementation
    #     # TODO: implement the client receive strategy correctlt later
    #     specific_ins = FedstdevStrategy.client_receive_strategy(client_ins)
    #     # NOTE: Current implementation assumes state persistence between download and upload calls.
    #     self._round = client_ins._round
    #     param_dict = client_ins.params

    #     match client_ins.request:
    #         case fed_t.RequestType.NULL:
    #             # Copy the model from the server
    #             self._model.load_state_dict(param_dict)
    #             return fed_t.RequestOutcome.COMPLETE
    #         case fed_t.RequestType.TRAIN:
    #             # Reset the optimizer
    #             self._model.load_state_dict(param_dict)
    #             self._optimizer = self.optim_partial(self._model.parameters())
    #             for seed, model in self._model_map.items():
    #                 model.load_state_dict(param_dict)
    #                 self._optimizer_map[seed] = self.optim_partial(model.parameters())
    #                 # self.res_man.log_parameters(self._model.state_dict(), 'post_agg', self._identifier, verbose=True)

    #             self._train_result = self.train()
    #             return fed_t.RequestOutcome.COMPLETE
    #         case fed_t.RequestType.EVAL:
    #             self._model.load_state_dict(param_dict)
    #             self._eval_result = self.eval()
    #             return fed_t.RequestOutcome.COMPLETE
    #         case fed_t.RequestType.RESET:
    #             # Reset the model to initial states
    #             self._model.load_state_dict(self._init_state_dict)
    #             return fed_t.RequestOutcome.COMPLETE
    #         case _:
    #             return fed_t.RequestOutcome.FAILED
        


    # def upload(self, request_type=fed_t.RequestType.NULL) -> fed_t.ClientResult1:
    #     # Upload the model back to the server

    #     match request_type:
    #         case fed_t.RequestType.TRAIN:
    #             self._model.to('cpu')
    #             strategy_ins = FedstdevStrategy.FedstdevIns(
    #                 client_params=self._model.state_dict(),
    #                 client_param_stds=self._get_parameter_std_dev()
    #             )
    #             client_result = FedstdevStrategy.client_send_strategy(strategy_ins, result=self._train_result)
    #             return client_result
    #         case fed_t.RequestType.EVAL:
    #             _result = self._eval_result
    #             return fed_t.ClientResult1(
    #                 params={},
    #                 result=_result)
    #         case _:
    #             _result = fed_t.Result(actor=self._cid,
    #                             _round=self._round,
    #                             size=self.__len__())
    #             self._model.to('cpu')
    #             client_result = fed_t.ClientResult1(
    #                 params=self.model.state_dict(keep_vars=False),
    #                 result=_result)
    #             return client_result
            

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
        
        return fed_t.Result(metrics=avg_metrics, size=sample_res.size, metadata=sample_res.metadata,event=sample_res.event, phase=sample_res.phase, _round=self._round, actor=self._identifier)
    
    def train_single_thread(self, train_ins: ClientInProtocol) -> fed_t.Result:
        self._model.load_state_dict(train_ins.server_params)
        self._optimizer = self.optim_partial(self._model.parameters())
        # Run an round on the client
        empty_grads = {p_key: torch.empty_like(param.data, device=self.cfg.device) for p_key, param in self._model.named_parameters()}

        self._cum_gradients_map = {seed: deepcopy(empty_grads) for seed in self._model_map.keys()}

        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.cfg.device)

        resume_epoch = 0
        out_result= fed_t.Result()
        out_result_dict: dict[int, fed_t.Result] = {}

        for seed, model in self._model_map.items():       
            # set optimizer parameters
            # optimizer: Optimizer = self.optim_partial(model.parameters())
            optimizer: Optimizer = self._optimizer_map[seed]

            model.train()
            model.to(self.cfg.device)
            # iterate over epochs and then on the batches
            for epoch in range(resume_epoch, resume_epoch + self.cfg.epochs):
                for inputs, targets in self.train_loader_map[seed]:

                    inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs: Tensor = model(inputs)
                    loss: Tensor = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    for p_key, param in model.named_parameters():
                        add_grad = 0 if param.grad is None else param.grad
                        self._cum_gradients_map[seed][p_key] = self._cum_gradients_map[seed].get(p_key, 0) + add_grad # type: ignore

                    # ic(f'Grad values in loop {seed}:')
                    # if not model.get_parameter(  'features.0.bias').grad is None:
                    #     ic(model.get_parameter(  'features.0.bias').grad[0])
                    #     ic(model.get_parameter('classifier.2.bias').grad[0])
                    #     ic(model.get_parameter(  'features.3.bias').grad[0])
                    #     ic(model.get_parameter('classifier.4.bias').grad[0])
                    # accumulate metrics
                    self.metric_mngr.track(loss.item(), outputs, targets)
                else:
                    out_result_dict[seed] = self.metric_mngr.aggregate(len(self.training_set), epoch)
                    self.metric_mngr.flush()

                self._epoch = epoch

          
        
            # ic(f'Grad values in loop per seed {seed}:')
            # if not model.get_parameter(  'features.0.bias').grad is None:
            #     ic(model.get_parameter(  'features.0.bias').grad[0])
            #     ic(model.get_parameter('classifier.2.bias').grad[0])
            #     ic(model.get_parameter(  'features.3.bias').grad[0])
            #     ic(model.get_parameter('classifier.4.bias').grad[0])

            # ic('In loop Param values:')
            # ic(model.get_parameter(  'features.0.bias').data[0])
            # ic(model.get_parameter('classifier.2.bias').data[0])
            # ic(model.get_parameter(  'features.3.bias').data[0])
            # ic(model.get_parameter('classifier.4.bias').data[0])

        self._compute_average_model_and_std(self._model_map)
        
        # ic('Param values:')
        # ic(self._model.get_parameter(  'features.0.bias').data[0])
        # ic(self._model.get_parameter('classifier.2.bias').data[0])
        # ic(self._model.get_parameter(  'features.3.bias').data[0])
        # ic(self._model.get_parameter('classifier.4.bias').data[0])

        # ic('Grad values:')
        # ic(self._model.get_parameter(  'features.0.bias').grad[0])
        # ic(self._model.get_parameter('classifier.2.bias').grad[0])
        # ic(self._model.get_parameter(  'features.3.bias').grad[0])
        # ic(self._model.get_parameter('classifier.4.bias').grad[0])
 
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
        self._model.to(self.cfg.device)
        out_result = fed_t.Result()
        # for seed, model in self._model_map.items():   
        with ThreadPoolExecutor(max_workers=len(self._model_map)) as exec:
            futures = {exec.submit(train_one_model, model, self.train_loader_map[seed], seed, self.cfg, self.optim_partial, self.criterion, deepcopy(self.metric_mngr)) for seed, model in self._model_map.items()}
            for fut in as_completed(futures):
                seed, model, out_result = fut.result()
                self._model_map[seed] = model
        
        self._compute_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')
        return out_result
 