import logging
from collections import OrderedDict
# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CosineSimilarity, Parameter
from torch.nn.utils import parameters_to_vector

from src.config import ClientConfig, CGSVConfig
from src.results.resultmanager import ClientResult
from .baseserver import BaseServer, BaseStrategy
logger = logging.getLogger(__name__)


class CgsvOptimizer(BaseStrategy):
    def __init__(self, params, client_ids, **kwargs):
        self.gamma = kwargs.get('gamma')
        self.lr = kwargs.get('lr')

        defaults = dict(lr=self.lr)
        super(CgsvOptimizer, self).__init__(params=params, defaults=defaults)

        self.alpha = kwargs.get('alpha')
        self.local_grad_norm = None
        self.server_grad_norm = None

        self._cos_sim = CosineSimilarity(dim=0)
        # NOTE: Differing in initialization here from paper as it leads to permanently zero gradients
        self._importance_coefficients = dict.fromkeys(client_ids, 1.0/len(client_ids))
        
    
    def _compute_cgsv(self, server_param, local_param):
        # self.local_grad_norm = self.server_para
        # print(server_param[0].dtype)
        # print(local_param[0].dtype)

        server_param_vec = parameters_to_vector(server_param)
        local_param_vec = parameters_to_vector(local_param)

        return self._cos_sim(server_param_vec, local_param_vec)

    def _update_coefficients(self, client_id, cgsv):
        self._importance_coefficients[client_id] = self.alpha * self._importance_coefficients[client_id] + (1 - self.alpha)* cgsv

        
    def _sparsify_gradients(self, client_ids):
        # TODO: Implement gradient sparsification for reward 
        pass

    def step(self, closure=None):
        # single step in a round of cgsv
        loss = None
        if closure is not None:
            loss = closure()

        # TODO: what to do if param groups are multiple
        
        for group in self.param_groups:
            # beta = group['momentum']
            for param in group['params']:
                # print("Param shape: ", param.shape)
                if param.grad is None:
                    continue
                # gradient
                delta = param.grad.data
                # FIXME: switch to an additive gradient with LR?
                # w = w - ∆w 
                param.data.sub_(delta)
        return loss

    def normalize_coefficients(self):
        total = 0
        for val in self._importance_coefficients.values():
            total += val
        for key, val in self._importance_coefficients.items():
            self._importance_coefficients[key] = val/total

        
    def accumulate(self, local_params_dict: OrderedDict[str, Parameter], client_id):
        # THis function is called per client. i.e. n clients means n calls
        # TODO: Rewrite this function to match gradient aggregate step
        # NOTE: Note that accumulate is called before step

        # print(f'Client ID as recvd: {client_id}')
        # NOTE: Currently supporting only one param group
        self._server_params = self.param_groups[0]['params']

        local_params = [param.data.float() for _, param in local_params_dict.items()]
        assert len(self._server_params) == len(local_params), f'Mismatch in parameter lengths'

        # print(local_params[0])
        # print(self._server_params[0].size())
        # print(self._server_params[0].norm())

        # exit(0)
        local_grads = []
        server_grads = []
        i = 0
        # print(len(self._server_params))
        for server_param, local_param in zip(self._server_params, local_params):
                i += 1
                local_delta = server_param - local_param

                norm = local_delta.norm() 
                if norm == 0:
                    logger.warning(f"CLIENT [{client_id}]: Got a zero norm!")
                    local_grad_norm = local_delta.mul(self.gamma)
                else:
                    local_grad_norm = local_delta.div(norm).mul(self.gamma)

                weighted_local_grad = local_grad_norm.mul(self._importance_coefficients[client_id])
                
                # server params grad is used as a buffer
                if server_param.grad is None:
                    server_param.grad = weighted_local_grad
                else:
                    server_param.grad.add_(weighted_local_grad)

                server_grads.append(server_param.grad.data)
                local_grads.append(local_grad_norm.data)

        cgsv = self._compute_cgsv(server_grads, local_grads)

        self._update_coefficients(client_id, cgsv)

class CgsvServer(BaseServer):
    name:str = 'CgsvServer'

    def __init__(self, cfg: CGSVConfig, *args, **kwargs):
        super(CgsvServer, self).__init__(cfg, *args, **kwargs)
        
        # self.server_optimizer = self._get_algorithm(self.model, lr=self.args.lr, gamma=self.args.gamma)
        self.round = 0
        self.cfg = cfg

        self.importance_coefficients = dict.fromkeys(self.clients, 0.0)

        self.server_optimizer: CgsvOptimizer = CgsvOptimizer(params=self.model.parameters(), client_ids=self.clients.keys(), lr=self.client_cfg.lr, gamma=self.cfg.gamma, alpha=self.cfg.alpha)

        # lr scheduler
        self.lr_scheduler = self.client_cfg.lr_scheduler(optimizer=self.server_optimizer)

 

    def _run_strategy(self, ids, train_results: ClientResult):
        # Calls client upload and server accumulate
        logger.debug(f'[{self.name}] [Round: {self.round:03}] Aggregate updated signals!')
        self.server_optimizer.zero_grad(set_to_none=True) # empty out buffer

        # accumulate weights
        for identifier in ids:
            local_params_dict = self.clients[identifier].upload()
            
            self.server_optimizer.accumulate(local_params_dict, identifier)

        self.server_optimizer.normalize_coefficients()
        self.server_optimizer.step() # update global model with the aggregated update
        self.lr_scheduler.step() # update learning rate
        logger.debug(f'[{self.name}] [Round: {self.round:03}] ...successfully aggregated into a new global model!')
