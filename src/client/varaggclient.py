from typing import Iterator, Tuple
from .baseclient import  *
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch import Generator
from torch.nn import Module, Parameter
from src.config import VaraggClientConfig
import logging
logger = logging.getLogger(__name__)

class VaraggClient(BaseClient):
    def __init__(self, cfg:VaraggClientConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        kwargs.get()

        self.cfg = cfg

        self.train_loader_map = self._create_shuffled_loaders(self.training_set, cfg.seeds)
        self._model_map: dict[int, Module] = {seed:copy.deepcopy(self._model) for seed in cfg.seeds}
        
        self._param_std = self._model._parameters

    def _create_shuffled_loaders(self, dataset:Dataset, seeds:list[int]) -> dict[int, DataLoader]:
        loader_dict = {}
        for seed in seeds:
            gen = Generator()
            gen.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
            loader_dict[seed] = DataLoader(dataset=dataset, sampler=sampler, batch_size=self.cfg.batch_size)

        return loader_dict
    

    def parameter_std_dev(self)->Iterator[Tuple[str, Parameter]]:
        for name, param in self._param_std.items():
            yield name, param


    def get_average_model_and_std(self, model_map:dict[int, Module]):

        for name, param in self._model.named_parameters():
            tmp_param_list = []
            for seed, model in model_map.items():
                tmp_param_list.append(model.get_parameter(name).data)

            stacked = torch.stack(tmp_param_list)
            std_, mean_ = torch.std_mean(stacked, dim=0)
            param.data.copy_(mean_.data)
            self._param_std[name] = std_


        # for name, param in self._model.named_parameters():
        #     ic(name, param.requires_grad)


    def train(self, return_model=False):
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
            for self._epoch in log_tqdm(range(self.cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
                for inputs, targets in self.train_loader_map[seed]:

                    inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                    model.zero_grad(set_to_none=True)

                    outputs:Tensor = model(inputs)
                    loss:Tensor = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # accumulate metrics
                    self.mm.track(loss.item(), outputs, targets)
                else:
                    out_result = self.mm.aggregate(len(self.training_set), self._epoch)
                    self.mm.flush()

        
        self.get_average_model_and_std(self._model_map)
        

        logger.info(f'CLIENT {self.id} Completed update')
        if return_model:
            return out_result, self._model.to('cpu')
        else:
            return out_result
        
    def reset_model(self) -> None:
        self._model = None
       