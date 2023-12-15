from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
import logging
from src.metrics.metricmanager import MetricManager, Result
from src.utils import log_tqdm
from src.config import ClientConfig
# from .varaggclient import VaraggClient

logger = logging.getLogger(__name__)


def model_eval_helper(model: Module, dataloader: DataLoader, cfg: ClientConfig, mm: MetricManager, round: int)->Result:
    # mm = MetricManager(cfg.eval_metrics, round, actor)
    mm._round = round
    model.eval()
    model.to(cfg.device)
    criterion = cfg.criterion

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss:Tensor = criterion(outputs, targets)
        mm.track(loss.item(), outputs, targets)
    else:
        result = mm.aggregate(len(dataloader.dataset), -1)
        mm.flush()
    return result


class BaseClient:
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, cfg: ClientConfig, id_seed: int, dataset: tuple, model: Module):
        self._identifier:str = f'{id_seed:04}' # potential to convert to hash
        # Always deepcopy the model_
        self._model: Module = deepcopy(model)

        self._init_state_dict: OrderedDict = model.state_dict()

        
        self._round = 0
        self._epoch = 0
        self._is_resumed = False
        self.cfg = deepcopy(cfg)
        self.training_set = dataset[0]
        self.test_set = dataset[1]

        
        self.mm = MetricManager(self.cfg.eval_metrics, self._round, actor=self._identifier)
        self.optim_partial: Optimizer = self.cfg.optimizer
        self.criterion = self.cfg.criterion

        self.train_loader = self._create_dataloader(self.training_set, shuffle=cfg.shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)
        self.optimizer: Optimizer = self.optim_partial(self._model.parameters())

        # self._debug_param: Tensor = None

    @property
    def id(self)->str:
        return self._identifier

    @property
    def model(self)-> Module:
        return self._model
    
    # @model.setter
    def set_model(self, model: Module):
        self._model = model
    
    def set_lr(self, lr:float) -> None:
        self.cfg.lr = lr

    @property
    def round(self)->int:
        return self._round
    @round.setter
    def round(self, value: int):
        self._round = value
    
    @property
    def epoch(self)->int:
        return self._epoch

    def reset_model(self) -> None:
        self._model.load_state_dict(self._init_state_dict)
        
    def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
        if self.cfg.batch_size == 0 :
            self.cfg.batch_size = len(self.training_set)
        return DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, shuffle=shuffle)
    

    def download(self, round:int, model_dict: OrderedDict):
        # Copy the model from the server
        self._round = round
        # self._model = copy.deepcopy(model)
        self._model.load_state_dict(model_dict)
        # print(f'Client {self.id} model: {id(self._model)}')

    def upload(self) -> OrderedDict:
        # Upload the model back to the server
        self._model.to('cpu')

        return self.model.state_dict(keep_vars=True)
        # return self._model.named_parameters()
    
    # Adding temp fix to return model under multiprocessing
    def train(self, return_model=False):
        # Run an round on the client
        # logger.info(f'CLIENT {self.id} Starting update')
        # mm = MetricManager(self.cfg.eval_metrics, self._round, actor=self._identifier)
        self.mm._round = self._round
        # ic('started_training')
        self._model.train()
        self._model.to(self.cfg.device)
        # ic(self.id, id(self._model.state_dict()))
        # set optimizer parameters again
        if not self._is_resumed:
            self.optimizer: Optimizer = self.optim_partial(self._model.parameters())
   
        # iterate over epochs and then on the batches
        for self._epoch in log_tqdm(range(self.cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                self._model.zero_grad(set_to_none=True)

                outputs: Tensor = self._model(inputs)
                loss: Tensor = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # ic(loss.item())

                # accumulate metrics
                self.mm.track(loss.item(), outputs, targets)
            else:
                out_result = self.mm.aggregate(len(self.training_set), self._epoch)
                self.mm.flush()
            if self._epoch + 1 % 10 == 0:
                self.save_checkpoint()

        self._epoch = 0

        if return_model:
            return out_result, self._model.to('cpu')
        else:
            return out_result


    @torch.no_grad()
    def evaluate(self):
        # Run evaluation on the client

        return model_eval_helper(self._model, self.test_loader, self.cfg, self.mm, self._round)

    def save_checkpoint(self):

        torch.save({
            'round': self._round,
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            }, f'client_ckpts/{self._identifier}/ckpt_r{self.round:003}_e{self._epoch:003}.pt')

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self._epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        logger.info(f'CLIENT {self.id} Loaded ckpt path: {ckpt_path}')
        self._round = checkpoint['round']



    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id:03} >'
    

