from flamby.datasets.fed_isic2019 import FedIsic2019
import logging
from src.config import DatasetModelSpec
from torch.utils.data import Subset, ConcatDataset
import src.common.typing as fed_t
import torch
logger = logging.getLogger(__name__)


def fetch_flamby_pooled(dataset_name: str, root: str)->tuple[Subset, Subset]:
    logger.debug(f'[DATA LOAD] Fetching dataset: {dataset_name.upper()}')

    match dataset_name:
        case "FedIsic2019":
            train_dataset = FedIsic2019(data_path=root, pooled=True, train=True)
            test_dataset = FedIsic2019(data_path=root, pooled=True, train=True)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return train_dataset, test_dataset

def custom_pooled(sharded_sets: fed_t.ClientDatasets_t)->tuple[Subset, Subset]:
    train_sets = []
    test_sets = []
    for train, test in sharded_sets:
        train_sets.append(train)
        test_sets.append(test)
    train_dataset = ConcatDataset(train_sets)
    test_dataset = ConcatDataset(test_sets)

    return train_dataset, test_dataset

def fetch_flamby_federated(dataset_name: str, root: str, num_splits: int)->tuple[fed_t.ClientDatasets_t, Subset]:
    logger.debug(f'[DATA LOAD] Fetching dataset: {dataset_name.upper()}')

    client_datasets = []
    match dataset_name:
        case "FedIsic2019":
            assert num_splits < 7, 'FedIsic2019 only supports upto 6 centres'
            for i in range(num_splits):
                train_dataset = FedIsic2019(center=i, data_path=root, pooled=False, train=True)
                test_dataset = FedIsic2019(center=i, data_path=root, pooled=False, train=False)
                client_datasets.append((train_dataset, test_dataset))
            pooled_test = FedIsic2019(data_path=root, pooled=True, train=False)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return client_datasets, pooled_test

def get_flamby_model_spec(dataset_name: str, root: str)-> DatasetModelSpec:
    match dataset_name:
        case "FedIsic2019":
            train_dataset = FedIsic2019(data_path=root, pooled=True)
            num_classes = len(torch.unique(torch.as_tensor(train_dataset.targets)))
            assert num_classes == 8
            model_spec = DatasetModelSpec(num_classes=num_classes, in_channels=3)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return model_spec