import torch
from torch.nn import Module
import inspect
import logging
import importlib
from hydra.utils import instantiate
from src.config import ModelConfig
logger = logging.getLogger(__name__)


#########################
# Weight initialization #
#########################
def init_weights(model: Module, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m: Module):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def init_model(cfg: ModelConfig) -> Module:
    # initialize the model class 
    model: Module = instantiate(cfg.model_spec)

    init_weights(model, cfg.init_type, cfg.init_gain)


    logger.info(f'[MODEL] Initialized model: {cfg.name}; (Initialization type: {cfg.init_type.upper()}))!')
    return model


#DEPRECATED: OVERRIDING THIS FUNCTION for lack of clarity here
def load_model(args: ModelConfig):
    # retrieve model skeleton
    model_class = importlib.import_module('src.models', package=__package__).__dict__[args.name]

    # get required model arguments
    required_args = inspect.getargspec(model_class)[0]

    # collect eneterd model arguments
    model_args = {}
    for argument in required_args:
        if argument == 'self': 
            continue
        model_args[argument] = getattr(args, argument)

    # get model instance
    model = model_class(**model_args)

    # adjust arguments if needed
    # if args.use_pt_model:
    #     args.num_embeddings = model.num_embeddings
    #     args.embedding_size = model.embedding_size
    #     args.num_hiddens = model.num_hiddens
    #     args.dropout = model.dropout
    
    # if args.name == 'LogReg':
    #     args.criterion = torch.nn.NLLLoss
    return model, args
