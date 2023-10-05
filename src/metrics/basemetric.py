from abc import ABC, abstractmethod
from torch import Tensor
from dataclasses import dataclass


# @dataclass
# class MetricConfig:
#     eval_metrics: list
#     fairness_metrics: list



class BaseMetric(ABC):
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


