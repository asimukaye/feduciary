from typing import Iterator, Tuple
import logging
from torch.nn import Parameter
import torch

from .fedavgserver import FedavgServer
from .baseserver import BaseServer, BaseOptimizer

logger = logging.getLogger(__name__)

class FedsgdOptimizer(BaseOptimizer):

    def __init__(self, params: Iterator[Parameter], **kwargs):
        self.lr = kwargs.get('lr')
        self.momentum = kwargs.get('momentum', 0.)
        defaults = dict(lr=self.lr, momentum=self.momentum)
        super(FedsgdOptimizer, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        # param groups is list of params. Initialized in the torch optFedimizer class 
        for group in self.param_groups:
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data
                if beta > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * grad
                    delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator: Iterator[Tuple[str, Parameter]]):

        for group in self.param_groups:

            for server_param, (name, local_param) in zip(group['params'], local_param_iterator):

                if server_param.grad is None: # NOTE: grad is used as buffer to accumulate local updates!
                    server_param.grad = server_param.data.sub(local_param.data).mul(mixing_coefficient)
                else:
                    server_param.grad.add_(server_param.data.sub(local_param.data).mul(mixing_coefficient))
        
class FedsgdServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedsgdServer, self).__init__(**kwargs)
