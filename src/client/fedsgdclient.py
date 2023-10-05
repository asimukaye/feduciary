from .fedavgclient import FedavgClient
from src.utils import MetricManager



class FedsgdClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedsgdClient, self).__init__(**kwargs)

    def update(self):
        # Different from FedAvg as this runs only one epoch 
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            for param in self.model.parameters():
                param.grad = None
            loss.backward()

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.training_set), 1)
        return mm.results
    