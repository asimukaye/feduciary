import os
import gc
import numpy as np
import logging
import torchtext
import torchvision.models
import transformers
from torch.utils import data
import concurrent.futures
from hydra.utils import instantiate
import feduciary.common.typing as fed_t
from feduciary.common.utils  import TqdmToLogger
from feduciary.datasets import *
from feduciary.split import get_client_datasets
import torchvision.transforms as tvt
logger = logging.getLogger(__name__)
from feduciary.config import DatasetConfig, TransformsConfig, DatasetModelSpec


def get_train_transform(cfg: TransformsConfig):
    if not cfg:
        train_list = [tvt.ToTensor()]
    else:
        train_list: list = instantiate(cfg.train_cfg)
        train_list.append(tvt.ToTensor())
        if cfg.normalize:
            train_list.append(instantiate(cfg.normalize))
    transform = tvt.Compose(train_list)
    return transform

def get_test_transform(cfg: TransformsConfig):
    if not cfg:
        tf_list = [tvt.ToTensor()]
    else:
        tf_list =[]
        if cfg.resize:
            tf_list.append(instantiate(cfg.resize))
        tf_list.append(tvt.ToTensor())
        if cfg.normalize:
            tf_list.append(instantiate(cfg.normalize))
    transform = tvt.Compose(tf_list)
    return transform
    
def load_vision_dataset(cfg: DatasetConfig) -> tuple[data.Dataset, data.Dataset, DatasetModelSpec]:
    # Deprecated in favor of load raw dataset
       
    transforms = [get_train_transform(cfg.transforms),
                get_test_transform(cfg.transforms)]
    raw_train, raw_test, model_spec = fetch_torchvision_dataset(dataset_name=cfg.name, root=cfg.data_path, transforms= transforms)

    if cfg.subsample:
        get_subset = lambda set, fraction: data.Subset(set, np.random.randint(0, len(set)-1, int(fraction * len(set)))) # type: ignore
        
        raw_train = get_subset(raw_train, cfg.subsample_fraction)
        raw_test = get_subset(raw_test, cfg.subsample_fraction)

    # adjust the number of clas ses in binary case
    if model_spec.num_classes == 2:
        raise NotImplementedError()

    return raw_test, raw_train, model_spec


def load_raw_dataset(cfg: DatasetConfig) -> tuple[data.Dataset, data.Dataset, DatasetModelSpec]:
    """Fetch and split requested datasets.
    
    Args:
        cfg: DatasetConfig"

    Returns: raw_train, raw_test, model_spec
    """

    transforms = [get_train_transform(cfg.transforms),
                  get_test_transform(cfg.transforms)]

    match cfg.dataset_family:
        case 'torchvision':
            raw_train, raw_test, model_spec = fetch_torchvision_dataset(dataset_name=cfg.name, root=cfg.data_path, transforms= transforms)
        case 'medmnist':
            raw_train, raw_test, model_spec = fetch_medmnist_dataset(dataset_name=cfg.name, root=cfg.data_path, transforms= transforms)
        case 'torchtext':
            raise NotImplementedError()
        case 'leaf':
            raise NotImplementedError()
        case 'flamby':
            logger.warn('Flamby datasets use their own transformations. Ignoring the transforms config.')

            raw_train, raw_test = fetch_flamby_pooled(dataset_name=cfg.name, root=cfg.data_path)
            model_spec = get_flamby_model_spec(dataset_name=cfg.name, root=cfg.data_path)
    
        case _:
            raise NotImplementedError()

    if cfg.subsample:
        get_subset = lambda set, fraction: data.Subset(set, np.random.randint(0, len(set)-1, int(fraction * len(set)))) # type: ignore
        
        raw_train = get_subset(raw_train, cfg.subsample_fraction)
        raw_test = get_subset(raw_test, cfg.subsample_fraction)

    # ic(raw_train[0][0].shape)
    # adjust the number of classes in binary case
    if model_spec.num_classes == 2:
        raise NotImplementedError()


    return raw_train, raw_test, model_spec


def pool_datasets(cfg: DatasetConfig, client_sets: list[fed_t.DatasetPair_t]):
    """Pools datasets from clients into a single dataset.

    Args:
        cfg: DatasetConfig
        client_sets: list[fed_t.DatasetPair_t]
            List of client datasets
    """
    if cfg.dataset_family == 'flamby':
        raise NotImplementedError()
    else:
        pooled_train = data.ConcatDataset([pair[0] for pair in client_sets])
        pooled_test = data.ConcatDataset([pair[1] for pair in client_sets])
    
    return pooled_train, pooled_test
        

def load_federated_dataset(cfg: DatasetConfig) -> tuple[ fed_t.ClientDatasets_t, data.Dataset, DatasetModelSpec]:
    if cfg.split_conf.split_type == 'defacto':
        # Use the originally provided splits
        if cfg.dataset_family == 'flamby':
            client_datasets, raw_test = fetch_flamby_federated(dataset_name=cfg.name, root=cfg.data_path, num_splits=cfg.split_conf.num_splits)
            model_spec = get_flamby_model_spec(dataset_name=cfg.name, root=cfg.data_path)
        else:
            raise NotImplementedError()
    else:
        # Artificially simulate a client split
        raw_train, raw_test, model_spec = load_raw_dataset(cfg)

        client_datasets = get_client_datasets(cfg.split_conf, raw_train, raw_test)

    return client_datasets, raw_test, model_spec


def subsample_dataset(dataset: data.Dataset, fraction: float):
    return data.Subset(dataset, np.random.randint(0, len(dataset)-1, int(fraction * len(dataset)))) # type: ignore
