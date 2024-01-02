from typing import Iterator, Tuple
from collections import OrderedDict
import logging
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch import Generator
from torch.nn import Module, Parameter

from .baseclient import  *
from src.config import FedstdevClientConfig
from src.common.utils import get_time
from src.common.typing import Result

logger = logging.getLogger(__name__)
 # NOTE: Multithreading causes atleast 3x slowdown for 2 epoch case. DO not use until necessary
def train_one_model(model:Module, dataloader: DataLoader, seed: int, cfg: ClientConfig, optim_partial, criterion, mm: MetricManager)->Tuple[int,Module,Result]:
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
                dataloader.dataset), _epoch)
            mm.flush()
    return (seed, model, out_result)

class FedstdevClient(BaseClient):
    # TODO: Fix the argument ordering structure
    def __init__(self, cfg: FedstdevClientConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = deepcopy(cfg)
        self.res_man: ResultManager = kwargs.get('res_man')

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
        # Make m copies of the model for independent train iterations
        self._model_map: dict[int, Module] = {seed: deepcopy(self._model) for seed in cfg.seeds}

        self._optimizer_map: dict[int, Optimizer] = {}
        for seed, model in self._model_map.items():
            self._optimizer_map[seed] = self.optim_partial(model.parameters())
        
        # self._param_std :OrderedDict[str, Parameter] = OrderedDict(self._model.named_parameters())

        self._param_std :OrderedDict[str, Parameter] = self._model.state_dict()
        self._grad_mu :dict[str, Tensor] = {p_key: param.grad for p_key, param in self._model.named_parameters()}
        self._none_grads = deepcopy(self._grad_mu)
        self.accumulated_gradients_map = {seed: deepcopy(self._none_grads) for seed in self._model_map.keys()}

        self._grad_std :dict[str, Tensor] = deepcopy(self._grad_mu)

        self.train = self.train_single_thread


    def _create_shuffled_loaders(self, dataset:Dataset, seeds:list[int]) -> dict[int, DataLoader]:
        loader_dict = {}
        # HACK: Hack to fix the batch size based on dataset size for imbalanced dataset
        # FIXME: IMPORTANT: Formalize the iteration to batch size correlation for imbalanced data experiments
        # n_iters = len(dataset)/self.cfg.batch_size
        # ic(self._identifier, n_iters)
        # ic(len(dataset))
        # ic(self.cfg.batch_size)

        # if self._identifier == '0000':
        #     self.cfg.batch_size =  int(self.cfg.batch_size/2.0)
        # new_iters = len(dataset)/self.cfg.batch_size
        # ic(new_iters)
        for seed in seeds:
            gen = Generator()
            gen.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
            loader_dict[seed] = DataLoader(dataset=dataset, sampler=sampler, batch_size=self.cfg.batch_size)

        return loader_dict
    
    def download(self, _round: int, model_dict: OrderedDict):
        super().download(_round, model_dict)
        # TODO: Debug logging to check the parameters on the client
        self.res_man.log_parameters(self._model.state_dict(), 'post_agg', self._identifier, verbose=True)
        for seed, model in self._model_map.items():
            model.load_state_dict(model_dict)
            # NOTE: Commenting below line as it seems to have no noticeable benefit. Might be needed for stateful optimizers
            # self._optimizer_map[seed] = self.optim_partial(model.parameters())


    #    TODO: Remove this if not necessary
    def upload(self) -> OrderedDict[str, Parameter]:
        # Upload the model back to the server
        self._model.to('cpu')
        return self._model.state_dict(keep_vars=False)
        # return OrderedDict(self._model.named_parameters())
    
    def get_parameter_std_dev(self)->OrderedDict[str, Tensor]:
        return deepcopy(self._param_std)
    
    def get_gradients_std_dev(self)->OrderedDict[str, Tensor]:
        return deepcopy(self._grad_std)
    
    def get_gradients_average(self)->OrderedDict[str, Tensor]:
        return deepcopy(self._grad_mu)

    def _compute_average_model_and_std(self, model_map: dict[int, Module]):

        for name, param in self._model.named_parameters():
            tmp_param_list = []
            tmp_grad_list = []

            for seed, model in model_map.items():
                individual_param = model.get_parameter(name)
                tmp_param_list.append(individual_param.data)
                # tmp_grad_list.append(individual_param.grad)
                tmp_grad_list.append(self.accumulated_gradients_map[seed][name])



            stacked = torch.stack(tmp_param_list)

            std_, mean_ = torch.std_mean(stacked, dim=0)

            stacked_grad = torch.stack(tmp_grad_list)
            std_grad_, mean_grad_ = torch.std_mean(stacked_grad, dim=0)


            param.data.copy_(mean_.data)
            # with get_time():
            param.grad = mean_grad_.data

            self._param_std[name] = std_.to('cpu')
            self._grad_mu[name] = mean_grad_
            self._grad_std[name] = std_grad_



        # Is this the bug???
        # Tie down the other models to the common model
        # with get_time():
        new_state_dict = self._model.state_dict()
        for seed, model in model_map.items():
            model.load_state_dict(new_state_dict)
            # self._optimizer_map[seed] = self.optim_partial(model.parameters())

        # for name, param in self._model.named_parameters():
        #     ic(name, param.requires_grad)

    def train_single_thread(self, return_model=False):
        # Run an round on the client
    
        self.accumulated_gradients_map = {seed: deepcopy(self._none_grads) for seed in self._model_map.keys()}
        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.cfg.device)


        # ic('Param values pre train:')
        # ic(self._model.get_parameter(  'features.0.bias').data[0])
        # ic(self._model.get_parameter('classifier.2.bias').data[0])
        # ic(self._model.get_parameter(  'features.3.bias').data[0])
        # ic(self._model.get_parameter('classifier.4.bias').data[0])

        # ic('Grad values pre train:')
        # if not self._model.get_parameter(  'features.0.bias').grad is None:
        #     ic(self._model.get_parameter(  'features.0.bias').grad[0])
        #     ic(self._model.get_parameter('classifier.2.bias').grad[0])
        #     ic(self._model.get_parameter(  'features.3.bias').grad[0])
        #     ic(self._model.get_parameter('classifier.4.bias').grad[0])


        for seed, model in self._model_map.items():       
            # set optimizer parameters
            # optimizer: Optimizer = self.optim_partial(model.parameters())
            optimizer: Optimizer = self._optimizer_map[seed]
            # ic(id(model))
            # ic(id(optimizer))

            model.train()
            model.to(self.cfg.device)
            # iterate over epochs and then on the batches
            # for self._epoch in log_tqdm(range(self.cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
            for epoch in range(self._start_epoch, self._start_epoch + self.cfg.epochs):
                for inputs, targets in self.train_loader_map[seed]:

                    inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs: Tensor = model(inputs)
                    loss: Tensor = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    for p_key, param in model.named_parameters():
                        if self.accumulated_gradients_map[seed][p_key] is None:
                            self.accumulated_gradients_map[seed][p_key] = param.grad
                        else:
                            self.accumulated_gradients_map[seed][p_key]+= param.grad

                    # ic(f'Grad values in loop {seed}:')
                    # if not model.get_parameter(  'features.0.bias').grad is None:
                    #     ic(model.get_parameter(  'features.0.bias').grad[0])
                    #     ic(model.get_parameter('classifier.2.bias').grad[0])
                    #     ic(model.get_parameter(  'features.3.bias').grad[0])
                    #     ic(model.get_parameter('classifier.4.bias').grad[0])
                    # accumulate metrics
                    self.metric_mngr.track(loss.item(), outputs, targets)
                else:
                    out_result = self.metric_mngr.aggregate(len(self.training_set), epoch)
                    self.metric_mngr.flush()

                self._epoch = epoch

            self.res_man.log_general_metric(out_result.metrics['loss'], 'train_loss', f'seed_{seed}', 'pre_avg')
            self.res_man.log_general_metric(out_result.metrics['acc1'], 'train_acc', f'seed_{seed}', 'pre_avg')
        
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
 
        self._start_epoch = epoch + 1

        logger.info(f'CLIENT {self.id} Completed update')

        # FIXME: OUt result is only for the last client
        if return_model:
            return out_result, self._model.to('cpu')
        else:
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

        # for seed, model in self._model_map.items():   
        with ThreadPoolExecutor(max_workers=len(self._model_map)) as exec:
            futures = {exec.submit(train_one_model, model, self.train_loader_map[seed], seed, self.cfg, self.optim_partial, self.criterion, deepcopy(self.metric_mngr)) for seed, model in self._model_map.items()}
            for fut in as_completed(futures):
                seed, model, out_result = fut.result()
                self._model_map[seed] = model

        
        self._compute_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')
        return out_result
 