import os
import torch


def check_simulator_config(cfg):
    pass

def check_client_config(cfg):
    pass

def check_model_config(cfg):
    pass

def check_algorithm_config(cfg):
    pass

def check_metric_config(cfg):
    pass

def check_optimizer_config(cfg):
    if cfg.optimizer not in torch.optim.__dict__.keys():
        err = f'`{cfg.optimizer}` is not a submodule of `torch.optim`... please check!'
        logger.exception(err)
        raise AssertionError(err)