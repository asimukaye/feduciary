# Helper functions to synchronize client i/o with server i/o

import feduciary.common.typing as fed_t
from feduciary.strategy.abcstrategy import *
from feduciary.results.resultmanager import ResultManager
from feduciary.client.abcclient import ABCClient

# def check_client_strategy_compatibility(client: ABCClient, strategy: ABCStrategy) -> None:
#     '''Check if client and strategy are compatible'''
#     client_requires = client.ClientInProtocol.__dict__.keys()
#     strategy_requires = strategy.StrategyIns.__dict__.keys()
#     if not client.supported_strategy(strategy):
#         raise ValueError(f'Client {client} does not support strategy {strategy}')
    

# def 