from src.client.baseflowerclient import BaseFlowerClient
from src.strategy.basestrategy import BaseStrategy
from src.simulator import *
import hydra
from omegaconf import OmegaConf

from src.config import Config, register_configs
from icecream import install, ic


def init_dataset_and_model(cfg: Config):
    '''Initialize the dataset and the model here'''
    # NOTE: THe model spec is being modified in place here.
    # TODO: Generalize this logic for all datasets
    server_set, train_set, dataset_model_spec  = load_vision_dataset(cfg.dataset)
    client_sets = get_client_datasets(cfg.dataset.split_conf, train_set)

    model_spec = cfg.model.model_spec
    if model_spec.in_channels is None:
        model_spec.in_channels = dataset_model_spec.in_channels
        logger.info(f'[MODEL CONFIG] Setting model in channels to {model_spec.in_channels}')
    else:
        logger.info(f'[MODEL CONFIG] Overriding model in channels to {model_spec.in_channels}')

    
    if model_spec.num_classes is None:
        model_spec.num_classes = dataset_model_spec.num_classes
        logger.info(f'[MODEL CONFIG] Setting model num classes to {model_spec.num_classes}')
    else:
        logger.info(f'[MODEL CONFIG] Overriding model num classes to {model_spec.num_classes}')

    cfg.model.model_spec = model_spec

    # self.model_instance: Module = instantiate(cfg.model.model_spec)
    model_instance = init_model(cfg.model)
    return server_set, client_sets, model_instance

@hydra.main(version_base=None, config_path="conf", config_name="flowerconfig")
def run_feduciary(cfg: Config):
    cfg: Config = OmegaConf.to_object(cfg)
    logger.debug((OmegaConf.to_yaml(cfg)))



    all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    
    clients: dict[str, BaseFlowerClient] = dict()

    strategy_partial = instantiate(cfg.server)

    ic(strategy_partial)

    server_dataset, client_datasets, model = init_dataset_and_model(cfg)

    
    res_man = ResultManager(cfg.simulator, logger=logger)
    strat: BaseFlowerServer = strategy_partial(model=model, dataset=server_dataset, clients= clients, strategy=BaseStrategy(model, cfg.strategy.cfg), result_manager=res_man)


    client_datasets_map = {}
    for cid, dataset in zip(all_client_ids,client_datasets):
        client_datasets_map[cid] = dataset
    

    def _client_fn(cid: str):
        client_partial: partial = instantiate(cfg.client)
        _model = deepcopy(model)
        _datasets = client_datasets_map[cid]

        return client_partial(client_id=cid,
                dataset=_datasets, model=_model)
    

    fl.simulation.start_simulation(
        strategy=strat,
        client_fn=_client_fn,
        clients_ids= all_client_ids,
        config=fl.server.ServerConfig(num_rounds=cfg.simulator.num_rounds),
        client_resources=cfg.simulator.flwr_resources,
    )

if __name__== "__main__":
    register_configs()
    run_feduciary()