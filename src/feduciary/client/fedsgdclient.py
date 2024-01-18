from .baseclient import BaseClient
from feduciary.metrics.metricmanager import MetricManager


class FedsgdClient(BaseClient):
    def __init__(self, **kwargs):
        super(FedsgdClient, self).__init__(**kwargs)

    def update(self):
        # Different from FedAvg as this runs only one epoch 
        mm = MetricManager(self.cfg.eval_metrics)
        self.model.train()
        self.model.to(self.cfg.device)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.cfg.device), targets.to(self.cfg.device)
            
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            for param in self.model.parameters():
                param.grad = None
            loss.backward()

            mm.track(loss.item(), outputs, targets)
        else:
            res = mm.aggregate(len(self.training_set), 1)
        return res
    