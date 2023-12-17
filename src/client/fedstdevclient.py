from typing import Iterator, Tuple
from collections import OrderedDict
from .baseclient import  *
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch import Generator
from torch.nn import Module, Parameter
from src.config import FedstdevClientConfig
import logging
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        self.cfg = cfg
        self.res_man: ResultManager = kwargs.get('res_man')

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
        # Make m copies of the model for independent train iterations
        self._model_map: dict[int, Module] = {seed: deepcopy(self._model) for seed in cfg.seeds}
        
        # self._param_std :OrderedDict[str, Parameter] = OrderedDict(self._model.named_parameters())

        self._param_std :OrderedDict[str, Parameter] = self._model.state_dict()
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
    
    def download(self, round: int, model_dict: OrderedDict):
        super().download(round, model_dict)
        # TODO: Debug logging to check the parameters on the client
        self.res_man.log_parameters(self._model.state_dict(), 'post_agg', self._identifier, verbose=True)
        for model in self._model_map.values():
            model.load_state_dict(model_dict)

    def upload(self) -> OrderedDict[str, Parameter]:
        # Upload the model back to the server
        self._model.to('cpu')
        return self._model.state_dict(keep_vars=True)
        # return OrderedDict(self._model.named_parameters())
    
    def parameter_std_dev(self)->OrderedDict[str, Tensor]:
        return deepcopy(self._param_std)

    def get_average_model_and_std(self, model_map: dict[int, Module]):

        for name, param in self._model.named_parameters():
            tmp_param_list = []
            for seed, model in model_map.items():
                tmp_param_list.append(model.get_parameter(name).data)

            stacked = torch.stack(tmp_param_list)
            std_, mean_ = torch.std_mean(stacked, dim=0)
            param.data.copy_(mean_.data)
            self._param_std[name] = std_.to('cpu')

        # for name, param in self._model.named_parameters():
        #     ic(name, param.requires_grad)

    def train_single_thread(self, return_model=False):
        # Run an round on the client
    
        self.mm._round = self._round
        self._model.train()
        self._model.to(self.cfg.device)
        

        for seed, model in self._model_map.items():       
            # set optimizer parameters
            optimizer: Optimizer = self.optim_partial(model.parameters())
            model.train()
            model.to(self.cfg.device)
            # iterate over epochs and then on the batches
            # for self._epoch in log_tqdm(range(self.cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
            for self._epoch in range(self.cfg.epochs):
                for inputs, targets in self.train_loader_map[seed]:

                    inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs: Tensor = model(inputs)
                    loss: Tensor = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # accumulate metrics
                    self.mm.track(loss.item(), outputs, targets)
                else:
                    out_result = self.mm.aggregate(len(self.training_set), self._epoch)
                    self.mm.flush()

        
        self.get_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')

        # FIXME: OUt result is only for the last client
        if return_model:
            return out_result, self._model.to('cpu')
        else:
            return out_result


   
    def train_multi_thread(self):
        # Run an round on the client
    
        self.mm._round = self._round
        self._model.train()
        self._model.to(self.cfg.device)

        # for seed, model in self._model_map.items():   
        with ThreadPoolExecutor(max_workers=len(self._model_map)) as exec:
            futures = {exec.submit(train_one_model, model, self.train_loader_map[seed], seed, self.cfg, self.optim_partial, self.criterion, deepcopy(self.mm)) for seed, model in self._model_map.items()}
            for fut in as_completed(futures):
                seed, model, out_result = fut.result()
                self._model_map[seed] = model

        
        self.get_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')
        return out_result
    # def reset_model(self) -> None:
    #     self._model = None
    
    