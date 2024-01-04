from abc import ABC, abstractmethod
from torch import Tensor


class BaseMetric(ABC):
    '''wrapper for computing metrics over a list of values'''
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred:Tensor, true:Tensor):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    @abstractmethod
    def summarize(self):
        pass


