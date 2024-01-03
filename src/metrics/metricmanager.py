from collections import defaultdict
# from typing import DefaultDict
from importlib import import_module
from .basemetric import BaseMetric
import src.common.typing as fed_t

# TODO: Consider merging with Result Manager Later
##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics: list[str], _round: int, actor: str):
        self.metric_funcs: dict[str, BaseMetric] = {
            name: import_module(f'.metricszoo', package=__package__).__dict__[name.title()]() for name in eval_metrics}
        self.figures = defaultdict(int) 
        self._result = fed_t.Result(_round=_round, actor=actor)
        self._round = _round
        self._actor = actor

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, epoch) -> fed_t.Result:
        # aggregate 
        avg_metrics = {name: module.summarize() for name, module in self.metric_funcs.items()}

        avg_metrics['loss'] = self.figures['loss'] / total_len

        self._result.metrics = avg_metrics
        self._result.metadata['epoch'] = epoch
        self._result.size = total_len
        self._result._round = self._round


        self.figures = defaultdict(int)
        return self._result

    def flush(self):
        self.figures = defaultdict(int)
        self._result = fed_t.Result(_round=self._round, actor=self._actor)
    
    # @property
    # def result(self):
    #     return self._result
