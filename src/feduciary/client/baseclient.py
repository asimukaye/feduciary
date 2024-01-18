from collections import OrderedDict
from dataclasses import dataclass, asdict
import functools
import typing as t
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
import logging
from feduciary.metrics.metricmanager import MetricManager
from feduciary.common.utils import log_tqdm
from feduciary.config import ClientConfig, TrainConfig
from feduciary.results.resultmanager import ResultManager
from feduciary.strategy.basestrategy import BaseStrategy
import feduciary.common.typing as fed_t
from feduciary.client.abcclient import ABCClient, model_eval_helper
logger = logging.getLogger(__name__)


@dataclass
class ClientInProtocol(t.Protocol):
    server_params: dict

@dataclass
class ClientOuts:
    client_params: fed_t.ActorParams_t
    data_size: int

class BaseClient(ABCClient):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self,
                 cfg: ClientConfig,
                 train_cfg: TrainConfig,
                 client_id: str,
                 dataset: tuple,
                 model: Module
                 ):
               
        # NOTE: the client object for Flower uses its own tmp directory. May cause side effects
        self._cid = client_id 
        self._model: Module = model
        self._init_state_dict: OrderedDict = OrderedDict(model.state_dict())

        # FIXME: Stateful clients will not work with multiprocessing
        self._round = int(0)
        self._epoch = int(0)
        self._start_epoch = cfg.start_epoch
        self._is_resumed = False


        #NOTE: IMPORTANT: Make sure to deepcopy the config in every child class
        self.cfg = deepcopy(cfg)
        self.train_cfg = deepcopy(train_cfg)

        self.training_set = dataset[0]
        self.test_set = dataset[1]

        
        self.metric_mngr = MetricManager(self.train_cfg.metric_cfg, self._round, actor=self._cid)
        # self.optim_partial: functools.partial = instantiate(self.train_cfg.optimizer)
        self.optim_partial: functools.partial = self.train_cfg.optimizer

        self.criterion = self.train_cfg.criterion

        self.train_loader = self._create_dataloader(self.training_set, shuffle=cfg.data_shuffle)
        
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)
        self._optimizer: Optimizer = self.optim_partial(self._model.parameters())

        self._train_result = fed_t.Result(actor=client_id)
        self._eval_result = fed_t.Result(actor=client_id)

    @property
    def id(self)->str:
        return self._cid

    @property
    def model(self)-> Module:
        return self._model
    
    # @model.setter
    def set_model(self, model: Module):
        self._model = model
    
    def set_lr(self, lr:float) -> None:
        self.train_cfg.lr = lr

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
        if self.train_cfg.batch_size == 0 :
            self.train_cfg.batch_size = len(self.training_set)
        return DataLoader(dataset=dataset, batch_size=self.train_cfg.batch_size, shuffle=shuffle)
        
    def unpack_train_input(self, client_ins: fed_t.ClientIns) -> ClientInProtocol:
        specific_ins = BaseStrategy.client_receive_strategy(client_ins)
        return specific_ins
    
    def pack_train_result(self, result: fed_t.Result) -> fed_t.ClientResult1:
        self._model.to('cpu')
        client_outs = ClientOuts(client_params=self.model.state_dict(keep_vars=False), data_size=result.size)
        general_res = BaseStrategy.client_send_strategy(client_outs, result)
        return general_res

    #TODO: Replicate packing unpacking for eval also 
    
    def download(self, client_ins: fed_t.ClientIns) -> fed_t.RequestOutcome:
        # Download initiates training. Is a blocking call without concurrency implementation
        specific_ins = self.unpack_train_input(client_ins=client_ins)
        # NOTE: Current implementation assumes state persistence between download and upload calls.
        self._round = client_ins._round

        param_dict = specific_ins.server_params

        match client_ins.request:
            case fed_t.RequestType.NULL:
                # Copy the model from the server
                self._model.load_state_dict(param_dict)
                return fed_t.RequestOutcome.COMPLETE
            case fed_t.RequestType.TRAIN:
                # Reset the optimizer
                self._train_result = self.train(specific_ins)
                return fed_t.RequestOutcome.COMPLETE
            case fed_t.RequestType.EVAL:
                self._model.load_state_dict(param_dict)
                self._eval_result = self.eval()
                return fed_t.RequestOutcome.COMPLETE
            case fed_t.RequestType.RESET:
                # Reset the model to initial states
                self._model.load_state_dict(self._init_state_dict)
                return fed_t.RequestOutcome.COMPLETE
            case _:
                return fed_t.RequestOutcome.FAILED
        

    def upload(self, request_type=fed_t.RequestType.NULL) -> fed_t.ClientResult1:
        # Upload the model back to the server
        match request_type:
            case fed_t.RequestType.TRAIN:
                _result = self._train_result
                return self.pack_train_result(_result)
            case fed_t.RequestType.EVAL:
                _result = self._eval_result
                return fed_t.ClientResult1(
                    params={},
                    result=_result)
            case _:
                _result = fed_t.Result(actor=self._cid,
                                _round=self._round,
                                size=self.__len__())
                self._model.to('cpu')
                client_result = fed_t.ClientResult1(
                    params=self.model.state_dict(keep_vars=False),
                    result=_result)
                return client_result
        
    
    # Adding temp fix to return model under multiprocessing
    def train(self, train_ins: ClientInProtocol) -> fed_t.Result:
        # Run a round on the client
        # logger.info(f'CLIENT {self.id} Starting update')
        # print('############# CWD: ##########', os.getcwd())
        self._model.load_state_dict(train_ins.server_params)
        self._optimizer = self.optim_partial(self._model.parameters())
        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.train_cfg.device)

        resume_epoch = 0
        out_result = fed_t.Result()
        # iterate over epochs and then on the batches
        for epoch in log_tqdm(range(resume_epoch, resume_epoch + self.train_cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.train_cfg.device), targets.to(self.train_cfg.device)

                self._model.zero_grad(set_to_none=True)

                outputs: Tensor = self._model(inputs)
                loss: Tensor = self.criterion(outputs, targets)
                loss.backward()

                self._optimizer.step()
                # ic(loss.item())

                # accumulate metrics
                self.metric_mngr.track(loss.item(), outputs, targets)
            else:
                # TODO: Current implementation has out result rewritten for every epoch. Fix to pass intermediate results.
                out_result = self.metric_mngr.aggregate(len(self.training_set), epoch)
                self.metric_mngr.flush()
            if epoch + 1 % 10 == 0:
                self.save_checkpoint(epoch)
            self._epoch = epoch

        # Helper code to retain the epochs if train is called without subsequent download calls
        self._start_epoch = self._epoch + 1
        # self._train_result = out_result

        return out_result


    @torch.no_grad()
    def eval(self, eval_ins = None) -> fed_t.Result:
        # Run evaluation on the client
        self._eval_result = model_eval_helper(self._model, self.test_loader, self.train_cfg, self.metric_mngr, self._round)
        return self._eval_result

    def save_checkpoint(self, epoch=0):

        torch.save({
            'round': self._round,
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict' : self._optimizer.state_dict(),
            }, f'client_ckpts/{self._cid}/ckpt_r{self._round:003}_e{epoch:003}.pt')

    def load_checkpoint(self, ckpt_path: str):
        # TODO: Fix the logic for loading checkpoints
        checkpoint = torch.load(ckpt_path)
        self._start_epoch = checkpoint['epoch']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        logger.info(f'CLIENT {self.id} Loaded ckpt path: {ckpt_path}')
        # TODO: Check if round is necessary or not
        self._round = checkpoint['round']
        self._is_resumed = True        

    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id:03} >'

