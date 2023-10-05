from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from src.metrics.metricmanager import MetricManager, Result
import inspect
import copy
import torch
from src.config import ClientConfig

from torch import Tensor



def model_eval_helper(model: Module, dataloader:DataLoader, cfg: ClientConfig, caller:str, round:int)->Result:
    mm = MetricManager(cfg.eval_metrics, round, caller)
    model.eval()
    model.to(cfg.device)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss = torch.nn.__dict__[cfg.criterion]()(outputs, targets)
        mm.track(loss.item(), outputs, targets)
    else:
        mm.aggregate(len(dataloader), -1)

    # log result
    return mm.result


class BaseClient:
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, id_seed: int, cfg: ClientConfig, dataset: tuple):
        self.__identifier:str = f'{id_seed:04}' # potential to convert to hash
        self.__model: Module = None
        
        self._round = 0
        self._epoch = 0
        self.cfg = cfg
        self.training_set = dataset[0]
        self.test_set = dataset[1]
        
        self.optim:Optimizer = torch.optim.__dict__[self.cfg.optimizer]
        self.criterion = torch.nn.__dict__[self.cfg.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=cfg.shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    @property
    def id(self)->str:
        return self.__identifier

    @property
    def model(self)->Module:
        return self.__model
    
    @property
    def round(self)->int:
        return self._round
    
    @property
    def epoch(self)->int:
        return self._epoch

    def reset_model(self) -> None:
        self.__model = None
        
    def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
        if self.cfg.batch_size == 0 :
            self.cfg.batch_size = len(self.training_set)
        return DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, shuffle=shuffle)
    

    def download(self, round, model):
        # Copy the model from the server
        self._round = round
        self.__model = copy.deepcopy(model)

    def upload(self):
        # Upload the model back to the server
        self.__model.to('cpu')
        return self.__model.named_parameters()
        
    def _refine_optim_args(self, args):
        # adding additional args
        #TODO: check what are the args being added
        # NOTE: This function captures all he optim args from global args and captures those which match the optim class
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        
        # print(refined_args)
        return refined_args


    def update(self):
        # Run an round on the client
        mm = MetricManager(self.cfg.eval_metrics, self._round, caller=self.__identifier)
        self.__model.train()
        self.__model.to(self.cfg.device)
        
        # set optimizer parameters
        optimizer:Optimizer = self.optim(self.__model.parameters(), **self._refine_optim_args(self.cfg))

        # iterate over epochs and then on the batches
        for self._epoch in range(self.cfg.epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                outputs:Tensor = self.__model(inputs)
                loss:Tensor = self.criterion()(outputs, targets)

                # NOTE: Is zeroing out the gradient necessary?
                # https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html#:~:text=It%20is%20beneficial%20to%20zero,backward()%20is%20called.

                self.__model.zero_grad(set_to_none=True)
                # for param in self.__model.parameters():
                #     param.grad = None

                loss.backward()
                optimizer.step()

                # print(outputs.size(), targets.size())
                # accumulate metrics
                mm.track(loss.item(), outputs, targets)
            else:
                # NOTE: This else is against a for loop. Seeing this for the first time here
                mm.aggregate(len(self.training_set), self._epoch)
                
        return mm.result

    @torch.inference_mode()
    def evaluate(self):
        # Run evaluation on the client

        return model_eval_helper(self.__model, self.test_loader, self.cfg, self.id, self._round)

    
    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id:03} >'
    

