import os
import sys
import torch

import logging
import numpy as np
import functools
import inspect
from tqdm import tqdm
from typing import Dict
from importlib import import_module
from collections import defaultdict, OrderedDict
from multiprocessing import Process

from time import perf_counter
from colorama import Fore, Style

logger = logging.getLogger(__name__)

def get_parameters_as_ndarray(net: torch.nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

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
            m_logger.debug(f'[{func.__name__}] Args: {argvals} | Attribs: {params} started')
            result = func(self, *args, **kwargs)
            m_logger.debug(f'[{func.__name__}] Completed!')
            return result
        return wrapper
    return decorator



class get_time:
    def __init__(self) -> None:
        curr_frame = inspect.currentframe()
        if curr_frame is not None:
            stack = curr_frame.f_back
            if stack is not None:
                self.name = f'{stack.f_code.co_filename.split("/")[-1] }:{Fore.LIGHTCYAN_EX}{stack.f_lineno}{Fore.RESET} in {stack.f_code.co_name}'

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = Style.DIM+f'{self.name} took {Style.NORMAL+Fore.LIGHTYELLOW_EX}{self.time:.6f}{Fore.RESET+Style.DIM} seconds'+Style.NORMAL
        print(self.readout)

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


"""
Helper functionality for interoperability with stdlib `logging`.
"""


from tqdm.std import tqdm as std_tqdm

LOGGER = logging.getLogger(__name__)


class log_tqdm(std_tqdm):  # pylint: disable=invalid-name
    """
    A version of tqdm that outputs the progress bar
    to Python logging instead of the console.
    The progress will be logged with the info level.

    Parameters
    ----------
    logger   : logging.Logger, optional
      Which logger to output to (default: logger.getLogger('tqdm.contrib.logging')).

    All other parameters are passed on to regular tqdm,
    with the following changed default:

    mininterval: 1
    bar_format: '{desc}{percentage:3.0f}%{r_bar}'
    desc: 'progress: '


    Example
    -------
    ```python
    import logging
    from time import sleep
    from tqdm.contrib.logging import log_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        for _ in log_tqdm(range(10), mininterval=1, logger=LOG):
            sleep(0.3)  # assume processing one item takes less than mininterval
    ```
    """
    def __init__(
            self,
            *args,
            # logger=None,  # type: logging.Logger
            # mininterval=1,  # type: float
            # bar_format='{desc}{percentage:3.0f}%{r_bar}',  # type: str
            # desc='progress: ',  # type: str
            **kwargs):
        if len(args) >= 2:
            # Note: Due to Python 2 compatibility, we can't declare additional
            #   keyword arguments in the signature.
            #   As a result, we could get (due to the defaults below):
            #     TypeError: __init__() got multiple values for argument 'desc'
            #   This will raise a more descriptive error message.
            #   Calling dummy init to avoid attribute errors when __del__ is called
            super(log_tqdm, self).__init__([], disable=True)
            raise ValueError('only iterable may be used as a positional argument')
        tqdm_kwargs = kwargs.copy()
        self._logger = tqdm_kwargs.pop('logger', None)
        self._mode = tqdm_kwargs.pop('mode', 'debug')
        tqdm_kwargs.setdefault('mininterval', 1)
        tqdm_kwargs.setdefault('bar_format', '{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]')
        tqdm_kwargs.setdefault('desc', 'progress: ')
        self._last_log_n = -1
        super(log_tqdm, self).__init__(*args, **tqdm_kwargs)

    def _get_logger(self):
        if self._logger is not None:
            return self._logger
        return LOGGER

    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            LOGGER.debug('ignoring message before any progress: %r', self.n)
            return
        if self.n == self._last_log_n:
            # avoid logging for the same progress multiple times
            LOGGER.debug('ignoring log message with same n: %r', self.n)
            return
        self._last_log_n = self.n
        if msg is None:
            msg = self.__str__()
        if not msg:
            LOGGER.debug('ignoring empty message: %r', msg)
            return
        if self._mode == 'debug':
            self._get_logger().debug('%s', msg)
        else:
            self._get_logger().info('%s', msg)