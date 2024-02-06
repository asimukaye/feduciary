from collections import OrderedDict
from dataclasses import dataclass, asdict
import functools
import typing as t
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
from hydra.utils import instantiate
import logging

from feduciary.metrics.metricmanager import MetricManager
from feduciary.common.utils import (log_tqdm,
                              unroll_param_keys,
                              roll_param_keys,
                              get_model_as_ndarray, 
                              convert_param_dict_to_ndarray,
                              convert_ndarrays_to_param_dict,
                              get_time)

from feduciary.config import ClientConfig, TrainConfig
from feduciary.client.abcclient import ABCClient, simple_evaluator
from feduciary.results.resultmanager import ResultManager
import feduciary.common.typing as fed_t
import pandas as pd
# DEFINE WHAT STRATEGY IS SUPPORTED HERE. This might be needed to support packing and unpacking
from feduciary.strategy.basestrategy import BaseStrategy
import flwr as fl

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)

logger = logging.getLogger(__name__)


def flatten_dict(nested: dict) -> dict:
    return pd.json_normalize(nested, sep='.').to_dict('records')[0]

def results_to_flower_fitres(client_res: fed_t.ClientResult) -> FitRes:

    ndarrays_updated = convert_param_dict_to_ndarray(client_res.params)
    # ndarrays_updated = convert_param_list_to_ndarray(client_res.params)

        # Serialize ndarray's into a Parameters object
    parameters_updated = fl.common.ndarrays_to_parameters(ndarrays_updated)
    # print(asdict(res))
    flattened_res = flatten_dict(asdict(client_res.result))
    
    flattened_res.update(roll_param_keys(list(client_res.params.keys())))

    # print("Flattened: ", flattened_res.keys())

    return FitRes(Status(code=Code.OK, message="Success"),
                    parameters=parameters_updated,
                    num_examples=client_res.result.size,
                    metrics=flattened_res)

def results_to_flower_evalres(res: fed_t.Result) -> EvaluateRes:
    flattened_res = flatten_dict(asdict(res))
    return EvaluateRes(Status(code=Code.OK, message="Success"),
                    loss=res.metrics['loss'],
                    num_examples=res.size,
                    metrics=flattened_res)

def flower_fitins_to_client_ins(ins: FitIns) -> fed_t.ClientIns:
    ndarrays_original = fl.common.parameters_to_ndarrays(ins.parameters)
    _param_keys = unroll_param_keys(ins.config) # type: ignore
    # param_dict = convert_ndarrays_to_param_lists(ndarrays_original)
    param_dict = convert_ndarrays_to_param_dict(_param_keys, ndarrays_original)
    _round = int(ins.config.pop('_round'))
    _request = fed_t.RequestType(ins.config.pop('_request'))
    assert _request == fed_t.RequestType.TRAIN
    return fed_t.ClientIns(params=param_dict, metadata=ins.config, _round=_round, request=_request)

def flower_evalins_to_client_ins(ins: EvaluateIns) -> fed_t.ClientIns:
    ndarrays_original = fl.common.parameters_to_ndarrays(ins.parameters)
    _param_keys = unroll_param_keys(ins.config) # type: ignore
    # param_dict = convert_ndarrays_to_param_lists(ndarrays_original)
    param_dict = convert_ndarrays_to_param_dict(_param_keys, ndarrays_original)
    _round = int(ins.config.pop('_round'))
    _request = fed_t.RequestType(ins.config.pop('_request'))
    assert _request == fed_t.RequestType.EVAL
    return fed_t.ClientIns(params=param_dict, metadata=ins.config, _round=_round, request=_request)

def set_parameters(net: Module, parameters: list[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


@dataclass
class ClientConfigProtocol(t.Protocol):
    optimizer: functools.partial
@dataclass
class ClientInProtocol(t.Protocol):
    server_params: dict

@dataclass
class ClientOuts:
    client_params: fed_t.ActorParams_t
    data_size: int

class BaseFlowerClient(ABCClient, fl.client.Client):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self,
                 cfg: ClientConfig,
                 train_cfg: TrainConfig,
                 client_id: str,
                 dataset: fed_t.DatasetPair_t,
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

    # @property
    # def round(self)-> int:
    #     return self._round
    # @round.setter
    # def round(self, value: int):
    #     self._round = value
    
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
    
    def pack_train_result(self, result: fed_t.Result) -> fed_t.ClientResult:
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
        

    def upload(self, request_type=fed_t.RequestType.NULL) -> fed_t.ClientResult:
        # Upload the model back to the server
        match request_type:
            case fed_t.RequestType.TRAIN:
                _result = self._train_result
                return self.pack_train_result(_result)
            case fed_t.RequestType.EVAL:
                _result = self._eval_result
                return fed_t.ClientResult(
                    params={},
                    result=_result)
            case _:
                _result = fed_t.Result(actor=self._cid,
                                _round=self._round,
                                size=self.__len__())
                self._model.to('cpu')
                client_result = fed_t.ClientResult(
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
        self._eval_result = simple_evaluator(self._model, self.test_loader, self.train_cfg, self.metric_mngr, self._round)
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
    

######## FLOWER FUNCTIONS ##########


    # def __init__(self, cid, net, trainloader, valloader):
    #     self.cid = cid
    #     self.net = net
    #     self.trainloader = trainloader
    #     self.valloader = valloader

    # TODO: Consider implementing properties at some later time
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Get parameters as a list of NumPy ndarray'sz
        # print("I AM BEING CALLED")
        ndarrays = get_model_as_ndarray(self._model)

        # Serialize ndarray's into a Parameters object
        parameters = fl.common.ndarrays_to_parameters(ndarrays)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        # print(f"[Client {self._cid}] fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's

        # Update local model, train, get updated parameters
        self._round = int(ins.config['_round'])
        # TODO: Need to use formal unpacking command here
        _metadata = ins.config
        # set_parameters(self._model, ndarrays_original)
        client_ins = flower_fitins_to_client_ins(ins)
        train_ins = self.unpack_train_input(client_ins=client_ins)

        res = self.train(train_ins)

        client_res = self.pack_train_result(res)

        fit_res = results_to_flower_fitres(client_res)

        # Build and return response

        return fit_res


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # print(f"[Client {self._cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = fl.common.parameters_to_ndarrays(parameters_original)
        self._round = int(ins.config['_round'])

        # TODO: Need to use formal unpacking command here
        _metadata = ins.config
        set_parameters(self._model, ndarrays_original)

        res = self.eval()
        eval_res = results_to_flower_evalres(res)

        return eval_res
    

