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

    # adjust the number of classes in binary case
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
        
        # case 'TinyImageNet': # 5) for other public datasets...
        #     #FIXME:
        #     raw_train, raw_test, temp_ = fetch_tinyimagenet(args=cfg, root=cfg.data_path, transforms=transforms)
        
        # case 'CINIC10':
        #     #FIXME:
        #     raw_train, raw_test, temp_ = fetch_cinic10(args=cfg, root=cfg.data_path, transforms=transforms)
        # case 'BeerReviews':
        #     aspect_type = {'A': 'aroma', 'L': 'look'}
        #     parsed_type = cfg.name[-1]
        #     if parsed_type in ['A', 'L']:
        #         aspect = aspect_type[parsed_type]
        #     else:
        #         err = '[DATA LOAD] Please check dataset name!'
        #         logger.exception(err)
        #         raise Exception(err)
        #     raw_train, raw_test, args = fetch_beerreviews(args=cfg, root=cfg.data_path, aspect=aspect, tokenizer=tokenizer)  
        # case 'Heart':
        #     split_map, client_datasets, args = fetch_heart(args=cfg, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)
    
        # case 'Adult':
        #     split_map, client_datasets, args = fetch_adult(args=cfg, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)
    
        # case 'Cover':
        #     split_map, client_datasets, args = fetch_cover(args=cfg, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)  
    
        # case 'GLEAM':
            # split_map, client_datasets, args = fetch_gleam(args=cfg, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction, seq_len=cfg.seq_len)
        case _:
            raise NotImplementedError()

    if cfg.subsample:
        get_subset = lambda set, fraction: data.Subset(set, np.random.randint(0, len(set)-1, int(fraction * len(set)))) # type: ignore
        
        raw_train = get_subset(raw_train, cfg.subsample_fraction)
        raw_test = get_subset(raw_test, cfg.subsample_fraction)

    ic(raw_train[0][0].shape)
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


# def get_text_data_stub(cfg: DatasetConfig):
#     tokenizer = None
#     # if cfg.use_model_tokenizer or cfg.use_pt_model:
#     #     assert cfg.model_name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'

#     if cfg.use_model_tokenizer:
#         assert cfg.model_name.lower() in transformers.models.__dict__.keys(), f'Please check if the model (`{cfg.model_name}`) is supported by `transformers` module!'
#         module = transformers.models.__dict__[f'{cfg.model_name.lower()}']
#         tokenizer = getattr(module, f'{cfg.model_name}Tokenizer').from_pretrained(TOKENIZER_STRINGS[cfg.model_name])
#         tokenizer = getattr(module, f'{cfg.model_name}Tokenizer').from_pretrained(TOKENIZER_STRINGS[cfg.model_name])

#     elif cfg.name in torchtext.datasets.__dict__.keys(): # 4) for downloadable datasets in `torchtext.datasets`...
#         raw_train, raw_test, args = fetch_torchtext_dataset(args=args, dataset_name=cfg.name, root=cfg.data_path, seq_len=cfg.seq_len,tokenizer=tokenizer) 

#     else: # x) for a dataset with no support yet or incorrectly entered...
#         err = f'[DATA LOAD] Dataset `{cfg.name}` is not supported or seems incorrectly entered... please check!'
#         logger.exception(err)
#         raise Exception(err)     
#     logger.info(f'[DATA LOAD] ...successfully fetched dataset!')
    
#     if cfg.dataset_family == 'torchtext':
#         raise NotImplementedError
#     else:
#         raise NotImplementedError

# def get_leaf_data_stub(cfg: DatasetConfig):
        
#     if cfg.name in ['FEMNIST', 'Shakespeare', 'Sent140', 'CelebA', 'Reddit']: # 1) for a special dataset - LEAF benchmark...
#         # define transform
#         if cfg.name in ['FEMNIST', 'CelebA']:
#             # check if `resize` is required
#             if cfg.resize is None:
#                 logger.info(f'[DATA LOAD] Dataset `{cfg.name}` may require `resize` argument; (recommended: `FEMNIST` - 28, `CelebA` - 84)!')
#             transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
#         elif cfg.name == 'Reddit':
#             cfg.rawsmpl = 1.0

#         # construct split hashmap, client datasets
#         # NOTE: for LEAF benchmark, values of `split_map` hashmap is not indices, but sample counts of tuple (training set, test set)!
#         split_map, client_datasets, args = fetch_leaf(
#             args=args,
#             dataset_name=cfg.name, 
#             root=cfg.data_path, 
#             seed=cfg.seed, 
#             raw_data_fraction=cfg.rawsmpl, 
#             test_fraction=cfg.test_fraction, 
#             transforms=transforms
#         )

#         # no global holdout set for LEAF
#         raw_test = None  
#     if cfg.dataset_family == 'leaf':
#         raise NotImplementedError
#     else:
#         raise NotImplementedError
    