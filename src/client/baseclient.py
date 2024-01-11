from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
import logging
from src.metrics.metricmanager import MetricManager
from src.common.utils import log_tqdm
from src.config import ClientConfig
from src.results.resultmanager import ResultManager
import src.common.typing as fed_t

logger = logging.getLogger(__name__)


def model_eval_helper(model: Module,
                      dataloader: DataLoader,
                      cfg: ClientConfig,
                      mm: MetricManager,
                      round: int)->fed_t.Result:
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
    def __init__(self, cfg: ClientConfig, client_id: str, dataset: tuple, model: Module, res_man: ResultManager = None):
        self._cid = client_id 
        # self._cid: str = f'{id_seed:04}' # potential to convert to hash
        self._model: Module = model
        self.res_man = res_man
        self._init_state_dict: dict = model.state_dict()

        self._round = 0
        self._epoch = 0
        self._start_epoch = 0

        self._is_resumed = False

        #NOTE: IMPORTANT: Make sure to deepcopy the config in every child class
        self.cfg = deepcopy(cfg)
        self.training_set = dataset[0]
        self.test_set = dataset[1]

        
        self.metric_mngr = MetricManager(self.cfg.metric_cfg, self._round, actor=self._cid)
        self.optim_partial = self.cfg.optimizer
        self.criterion = self.cfg.criterion

        self.train_loader = self._create_dataloader(self.training_set, shuffle=cfg.shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)
        self._optimizer: Optimizer = self.optim_partial(self._model.parameters())

        # self._debug_param: Tensor = None

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
    

    def download(self, round:int, model_dict: dict[str, Parameter]):
        # Copy the model from the server
        self._round = round
        # Reset the epochs once a new model is supplied
        self._start_epoch = 0
        # self._model = copy.deepcopy(model)
        self._model.load_state_dict(model_dict)
        # ic('post download')
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
        # Reset the optimizer
        self._optimizer = self.optim_partial(self._model.parameters())
        # print(f'Client {self.id} model: {id(self._model)}')

    def upload(self) -> OrderedDict:
        # Upload the model back to the server
        self._model.to('cpu')
        # ic('pre upload')
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

        return self.model.state_dict(keep_vars=False)
        # return self._model.named_parameters()
    
    # Adding temp fix to return model under multiprocessing
    def train(self, return_model=False):
        # Run a round on the client
        # logger.info(f'CLIENT {self.id} Starting update')
        self.metric_mngr._round = self._round
        self._model.train()
        self._model.to(self.cfg.device)

        # ic('Grad values pre train:')
        # if not self._model.get_parameter(  'features.0.bias').grad is None:
        #     ic(self._model.get_parameter(  'features.0.bias').grad[0])
        #     ic(self._model.get_parameter('classifier.2.bias').grad[0])
        #     ic(self._model.get_parameter(  'features.3.bias').grad[0])
        #     ic(self._model.get_parameter('classifier.4.bias').grad[0])
        # set optimizer parameters again
        # if not self._is_resumed:
        #     self._optimizer: Optimizer = self.optim_partial(self._model.parameters())
   
        # iterate over epochs and then on the batches
        for epoch in log_tqdm(range(self._start_epoch, self._start_epoch + self.cfg.epochs), logger=logger, desc=f'Client {self.id} updating: '):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)

                self._model.zero_grad(set_to_none=True)


                outputs: Tensor = self._model(inputs)
                loss: Tensor = self.criterion(outputs, targets)
                loss.backward()
                self._optimizer.step()
                # ic(loss.item())

                # accumulate metrics
                self.metric_mngr.track(loss.item(), outputs, targets)
            else:
                out_result = self.metric_mngr.aggregate(len(self.training_set), epoch)
                self.metric_mngr.flush()
            if epoch + 1 % 10 == 0:
                self.save_checkpoint(epoch)
            self._epoch = epoch

        # Helper code to retain the epochs if train is called without subsequent download calls
        self._start_epoch = epoch + 1


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
 


        if return_model:
            return out_result, self._model.to('cpu')
        else:
            return out_result


    @torch.no_grad()
    def eval(self):
        # Run evaluation on the client

        return model_eval_helper(self._model, self.test_loader, self.cfg, self.metric_mngr, self._round)

    def save_checkpoint(self, epoch=0):

        torch.save({
            'round': self._round,
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict' : self._optimizer.state_dict(),
            }, f'client_ckpts/{self._cid}/ckpt_r{self.round:003}_e{epoch:003}.pt')

    def load_checkpoint(self, ckpt_path: str):
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
    

