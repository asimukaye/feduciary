import copy
import torch
import inspect

from .baseclient import BaseClient
from src import MetricManager


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    def _refine_optim_args(self, args):
        # adding additional args
        #TODO: check what are the args being added
        # NOTE: This function captures all he optim args from global args and captures those which match the optim class
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        # Run an round on the client
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        # set optimizer parameters
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        # iterate over epochs and then on the batches
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                # NOTE: Is zeroing out the gradient necessary?
                # https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html#:~:text=It%20is%20beneficial%20to%20zero,backward()%20is%20called.

                self.model.zero_grad(set_to_none=True)
                # for param in self.model.parameters():
                #     param.grad = None

                loss.backward()
                optimizer.step()

                # accumulate metrics
                mm.track(loss.item(), outputs, targets)
            else:
                # NOTE: This else is against a for loop. Seeing this for the first time here
                mm.aggregate(len(self.training_set), e + 1)
                
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        # Run evaluation on the client

        if self.args._train_only: # `args.test_fraction` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        # Copy the model from the server
        self.model = copy.deepcopy(model)

    def upload(self):
        # Upload the model back to the server
        self.model.to('cpu')
        return self.model.named_parameters()
        
    
    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
