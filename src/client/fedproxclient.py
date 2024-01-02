import copy
from typing import Protocol
from dataclasses import dataclass
from .baseclient import BaseClient
from src.metrics.metricmanager import MetricManager

@dataclass
class FedproxClientCfg(Protocol):
    device: str
    eval_metrics: list
    mu: float

# FIXME: Broken Implementation
class FedproxClient(BaseClient):
    def __init__(self, cfg: FedproxClientCfg, **kwargs):
        self.cfg = cfg
        super(FedproxClient, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.cfg.eval_metrics)
        self.model.train()  
        self.model.to(self.cfg.device)
        
        # NOTE: Different from FedAvg here. Uses a global model as reference.
        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): 
            param.requires_grad = False

        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.cfg))
        for e in range(self.cfg.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)
                
                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                prox = 0.
                for name, param in self.model.named_parameters():
                    prox += (param - global_model.get_parameter(name)).norm(2)
                loss += self.cfg.mu * (0.5 * prox)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()
                
                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        return mm.results
    