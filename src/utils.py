import os
import sys
import torch
import random
import logging
import numpy as np
import functools
import inspect
from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process

logger = logging.getLogger(__name__)

def log_instance(attrs:list=[], m_logger=logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            argvals= {}
            params= {}
            # for note in argread:
            #     if note in inspect.getfullargspec(func).args:
            #         # argvals[note] =
            #         print(args)
            for attr in attrs:
                params[attr]= getattr(self, attr)
            m_logger.info(f'[{func.__name__}] Args: {argvals} | Attribs: {params} started')
            result = func(self, *args, **kwargs)
            m_logger.info(f'[{func.__name__}] Completed!')
            return result
        return wrapper
    return decorator


#########################
# Argparser Restriction #
#########################
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end


###############
# TensorBaord #
###############
class TensorBoardRunner:
    def __init__(self, path, host, port):
        logger.info('[TENSORBOARD] Start TensorBoard process!')
        self.server = TensorboardServer(path, host, port)
        self.server.start()
        self.daemon = True
         
    def finalize(self):
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        self.server.pkill()
        logger.info('[TENSORBOARD] ...finished TensorBoard process!')
        
    def interrupt(self):
        self.server.pkill()
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        logger.info('[TENSORBOARD] ...interrupted; killed all TensorBoard processes!')

class TensorboardServer(Process):
    def __init__(self, path, host, port):
        super().__init__()
        self.os_name = os.name
        self.path = str(path)
        self.host = host
        self.port = port
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --port {self.port} 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --port {self.port} >/dev/null 2>&1')
        else:
            err = f'Current OS ({self.os_name}) is not supported!'
            logger.exception(err)
            raise Exception(err)
    
    def pkill(self):
        if self.os_name == 'nt':
            os.system(f'taskkill /IM "tensorboard.exe" /F')
        elif self.os_name == 'posix':
            os.system('pgrep -f tensorboard | xargs kill -9')

###############
# tqdm add-on #
###############
class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, 
    mininterval=0.1, 
    bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
    desc=None, 
    **kwargs
    ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))


#####################
# Arguments checker #
#####################
# TODO: Incorporate this into config checks later
def check_args(args):
    # check optimizer wrt torch optimizers
    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` is not a submodule of `torch.optim`... please check!'
        logger.exception(err)
        raise AssertionError(err)
    
    # check criterion wrt torch nn criterions
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` is not a submodule of `torch.nn`... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # TODO: this is an algo specific check
    # check algorithm, only for fedsgd
    if args.algorithm == 'fedsgd':
        args.E = 1
        args.B = 0

    # check lr step
    if args.lr_decay_step >= args.R:
        err = f'step size for learning rate decay (`{args.lr_decay_step}`) should be smaller than total round (`{args.R}`)... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check train only mode
    if args.test_fraction == 0:
        args._train_only = True
    else:
        args._train_only = False

    # check compatibility of evaluation metrics
    if hasattr(args, 'num_classes'):
        # Classification metrics check

        if args.num_classes > 2:
            if ('auprc' or 'youdenj') in args.eval_metrics:
                err = f'some metrics (`auprc`, `youdenj`) are not compatible with multi-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if 'acc5' in args.eval_metrics:
                err = f'Top5 accruacy (`acc5`) is not compatible with binary-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)

        if ('mse' or 'mae' or 'mape' or 'rmse' or 'r2' or 'd2') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a classification task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)
    else:
        # Regression metrics check
        if ('acc1' or 'acc5' or 'auroc' or 'auprc' or 'youdenj' or 'f1' or 'precision' or 'recall' or 'seqacc') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a regression task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)

    # print welcome message
    logger.info('[CONFIG] List up configurations...')
    for arg in vars(args):
        logger.info(f'[CONFIG] - {str(arg).upper()}: {getattr(args, arg)}')
    else:
        print('')
    return args

#####################
# BCEWithLogitsLoss #
#####################
class NoPainBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Native `torch.nn.BCEWithLogitsLoss` requires squeezed logits shape and targets with float dtype.
    """
    def __init__(self, **kwargs):
        super(NoPainBCEWithLogitsLoss, self).__init__(**kwargs)

    def forward(self, inputs, targets):
        return super(NoPainBCEWithLogitsLoss, self).forward(
            torch.atleast_1d(inputs.squeeze()), 
            torch.atleast_1d(targets).float()
        )

# NOTE: overriding a torch class implementation
torch.nn.BCEWithLogitsLoss = NoPainBCEWithLogitsLoss
