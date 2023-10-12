from collections import defaultdict
# from typing import DefaultDict
from importlib import import_module

from src.results.resultmanager import Result


##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics: list[str], round: int, caller: str):
        self.metric_funcs = {
            name: import_module(f'.metricszoo', package=__package__).__dict__[name.title()]() for name in eval_metrics}
        self.figures = defaultdict(int) 
        self._result = Result(round=round, caller=caller)
        self._round = round
        self._caller = caller

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, epoch):
        # aggregate 
        avg_metrics = {name: module.summarize() for name, module in self.metric_funcs.items()}

        avg_metrics['loss'] = self.figures['loss'] / total_len

        self._result.metrics = avg_metrics
        self._result.epoch = epoch
        self._result.size = total_len
        self._result.round = self._round


        self.figures = defaultdict(int)
        return self._result

    def flush(self):
        self.figures = defaultdict(int)
        self._result = Result(round=self._round, caller=self._caller)
    
    # @property
    # def result(self):
    #     return self._result
