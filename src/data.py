import os
import gc
import numpy as np
import logging
import torchtext
import torchvision
import transformers
from torch.utils import data
import concurrent.futures
from hydra.utils import instantiate
import src.common.typing as fed_t
from src.common.utils  import TqdmToLogger
from src.datasets import *
from src.split import get_client_datasets
import torchvision.transforms as tvt
logger = logging.getLogger(__name__)
from src.config import DatasetConfig, TransformsConfig, DatasetModelSpec


def get_train_transform(cfg: TransformsConfig):
    train_list: list = instantiate(cfg.train_cfg)
    train_list.append(tvt.ToTensor())
    if cfg.normalize:
        train_list.append(instantiate(cfg.normalize))
    transform = tvt.Compose(train_list)
    return transform

def get_test_transform(cfg: TransformsConfig):
    tf_list =[]
    if cfg.resize:
        tf_list.append(instantiate(cfg.resize))
    tf_list.append(tvt.ToTensor())
    if cfg.normalize:
        tf_list.append(instantiate(cfg.normalize))
    transform = tvt.Compose(tf_list)
    return transform
    
def load_vision_dataset(cfg: DatasetConfig) -> tuple[data.Subset, data.Subset, DatasetModelSpec]:
       
    transforms = [get_train_transform(cfg.transforms), get_test_transform(cfg.transforms)]
    raw_train, raw_test, model_spec = fetch_torchvision_dataset(dataset_name=cfg.name, root=cfg.data_path, transforms= transforms)

    if cfg.subsample:
        get_subset = lambda set, fraction: data.Subset(set, np.random.randint(0, len(set)-1, int(fraction * len(set))))
        
        raw_train = get_subset(raw_train, cfg.subsample_fraction)
        raw_test = get_subset(raw_test, cfg.subsample_fraction)

    # adjust the number of classes in binary case
    if model_spec.num_classes == 2:
        raise NotImplementedError()

    return raw_test, raw_train, model_spec


def load_raw_dataset(cfg: DatasetConfig) -> tuple[data.Subset, data.Subset, DatasetModelSpec]:
    """Fetch and split requested datasets.
    
    Args:
        cfg: DatasetConfig"

    Returns: raw_test, raw_train, model_spec
    """

    transforms = [get_train_transform(cfg.transforms), get_test_transform(cfg.transforms)]

    match cfg.dataset_family:
        case 'torchvision':
            raw_train, raw_test, model_spec = fetch_torchvision_dataset(dataset_name=cfg.name, root=cfg.data_path, transforms= transforms)
        case 'torchtext':
            raise NotImplementedError()
        case 'leaf':
            raise NotImplementedError()
        case 'flamby':
            logger.warn('Flamby datasets use their own transformations. Ignoring the transforms config.')
            # if cfg.split_conf.split_type == 'defacto':
            #     temp = fetch_flamby_federated(dataset_name=cfg.name, root=cfg.data_path, num_splits=cfg.split_conf.num_splits)
              #FIXME:
            # else:
            #     raw_train, raw_test = fetch_flamby_pooled(dataset_name=cfg.name, root=cfg.data_path)
            # model_spec = get_flamby_model_spec(dataset_name=cfg.name, root=cfg.data_path)
            raise NotImplementedError()

        case _:
            raise NotImplementedError()

    if cfg.subsample:
        get_subset = lambda set, fraction: data.Subset(set, np.random.randint(0, len(set)-1, int(fraction * len(set))))
        
        raw_train = get_subset(raw_train, cfg.subsample_fraction)
        raw_test = get_subset(raw_test, cfg.subsample_fraction)

    # adjust the number of classes in binary case
    if model_spec.num_classes == 2:
        raise NotImplementedError()


    return raw_train, raw_test, model_spec

def load_federated_dataset(cfg: DatasetConfig) -> tuple[ fed_t.ClientDatasets_t, data.Subset, DatasetModelSpec]:
    #FIXME: This is a hack to get the model spec
    if cfg.split_conf.split_type == 'defacto':
        if cfg.dataset_family == 'flamby':
            client_datasets, raw_test = fetch_flamby_federated(dataset_name=cfg.name, root=cfg.data_path, num_splits=cfg.split_conf.num_splits)
            model_spec = get_flamby_model_spec(dataset_name=cfg.name, root=cfg.data_path)
        else:
            raise NotImplementedError()
    else:
        raw_train, raw_test, model_spec = load_raw_dataset(cfg)

        client_datasets = get_client_datasets(cfg.split_conf, raw_train, raw_test)

    return client_datasets, raw_test, model_spec


def subsample_dataset(dataset: data.Dataset, fraction: float):
    return data.Subset(dataset, np.random.randint(0, len(dataset)-1, int(fraction * len(dataset))))



# TODO: Remove me Broken usage as of now
# DEPRECATED
# def get_transform(cfg: TransformsConfig, train=False):
#     transform = tvt.Compose(
#         [
#             tvt.Resize((cfg.resize, cfg.resize)) if cfg.resize is not None else tvt.Lambda(lambda x: x),
#             tvt.RandomRotation(cfg.randrot) if (cfg.randrot is not None and train) else tvt.Lambda(lambda x: x),
#             tvt.RandomHorizontalFlip(cfg.randhf) if (cfg.randhf is not None and train) else tvt.Lambda(lambda x: x),
#             tvt.RandomVerticalFlip(cfg.randvf) if (cfg.randvf is not None and train) else tvt.Lambda(lambda x: x),
#             tvt.ColorJitter(brightness=cfg.randjit, contrast=cfg.randjit, saturation=cfg.randjit, hue=cfg.randjit) if (cfg.randjit is not None and train) else tvt.Lambda(lambda x: x),
#             tvt.ToTensor(),
#             tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.imnorm else tvt.Lambda(lambda x: x)
#         ]
#     )
#     return transform


# DEPRECATED: Kept for code retrieval only

# def load_dataset(cfg:DatasetConfig):
#     """Fetch and split requested datasets.
    
#     Args:
#         args: arguments
        
#     Returns:
#         split_map: {client ID: [assigned sample indices]}
#             ex) {0: [indices_1], 1: [indices_2], ... , K: [indices_K]}
#         server_testset: (optional) holdout dataset located at the central server, 
#         client datasets: [(local training set, local test set)]
#             ex) [tuple(local_training_set[indices_1], local_test_set[indices_1]), tuple(local_training_set[indices_2], local_test_set[indices_2]), ...]

#     """
#     TOKENIZER_STRINGS = {
#         'DistilBert': 'distilbert-base-uncased',
#         'SqueezeBert': 'squeezebert/squeezebert-uncased',
#         'MobileBert': 'google/mobilebert-uncased'
#     } 
    
#     # error manager
#     def _check_and_raise_error(entered, targeted, msg, eq=True):
#         if eq:
#             if entered == targeted: # raise error if eq(==) condition meets
#                 err = f'[{cfg.name.upper()}] `{entered}` {msg} is not supported for this dataset!'
#                 logger.exception(err)
#                 raise AssertionError(err)
#         else:
#             if entered != targeted: # raise error if neq(!=) condition meets
#                 err = f'[{cfg.name.upper()}] `{targeted}` {msg} is only supported for this dataset!'
#                 logger.exception(err)
#                 raise AssertionError(err)

#     # method to get transformation chain

#     def _get_transform(args, train=False):
#         transform = tvt.Compose(
#             [
#                 tvt.Resize((cfg.resize, cfg.resize)) if cfg.resize is not None else tvt.Lambda(lambda x: x),
#                 tvt.RandomRotation(cfg.randrot) if (cfg.randrot is not None and train) else tvt.Lambda(lambda x: x),
#                 tvt.RandomHorizontalFlip(cfg.randhf) if (cfg.randhf is not None and train) else tvt.Lambda(lambda x: x),
#                 tvt.RandomVerticalFlip(cfg.randvf) if (cfg.randvf is not None and train) else tvt.Lambda(lambda x: x),
#                 tvt.ColorJitter(brightness=cfg.randjit, contrast=cfg.randjit, saturation=cfg.randjit, hue=cfg.randjit) if (cfg.randjit is not None and train) else tvt.Lambda(lambda x: x),
#                 tvt.ToTensor(),
#                 tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.imnorm else tvt.Lambda(lambda x: x)
#             ]
#         )
#         return transform

#     # method to construct per-client dataset
#     # def _construct_dataset(raw_train, idx, sample_indices) ->(SubsetWrapper, SubsetWrapper):
#     #     subset = data.Subset(raw_train, sample_indices)
#     #     test_size = int(len(subset) * cfg.test_fraction)
#     #     training_set, test_set = data.random_split(subset, [len(subset) - test_size, test_size])
#     #     traininig_set = SubsetWrapper(training_set, f'< {str(idx).zfill(8)} > (train)')
#     #     test_set = SubsetWrapper(test_set, f'< {str(idx).zfill(8)} > (test)')
#     #     return (traininig_set, test_set)
    
#     #################
#     # base settings #
#     #################
#     # required intermediate outputs
#     raw_train, raw_test = None, None

#     # required outputs
#     split_map, client_datasets = None, None
    
#     # optional argument for data transforms
#     transforms = [None, None]
    
#     ####################
#     # for text dataset #
#     ####################
#     tokenizer = None
#     # if cfg.use_model_tokenizer or cfg.use_pt_model:
#     #     assert cfg.model_name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'

#     if cfg.use_model_tokenizer:
#         assert cfg.model_name.lower() in transformers.models.__dict__.keys(), f'Please check if the model (`{cfg.model_name}`) is supported by `transformers` module!'
#         module = transformers.models.__dict__[f'{cfg.model_name.lower()}']
#         tokenizer = getattr(module, f'{cfg.model_name}Tokenizer').from_pretrained(TOKENIZER_STRINGS[cfg.model_name])

#     #################
#     # fetch dataset #
#     #################
#     logger.info(f'[DATA LOAD] Fetch dataset!')
    
#     if cfg.name in ['FEMNIST', 'Shakespeare', 'Sent140', 'CelebA', 'Reddit']: # 1) for a special dataset - LEAF benchmark...
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario', False)
#         _check_and_raise_error(cfg.eval_type, 'local', 'evaluation type', False)
         
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

#     elif cfg.name in torchvision.datasets.__dict__.keys(): # 3) for downloadable datasets in `torchvision.datasets`...
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario')
#         transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
#         raw_train, raw_test, args = fetch_torchvision_dataset(args=args, dataset_name=cfg.name, root=cfg.data_path, transforms=transforms)
        
#     elif cfg.name in torchtext.datasets.__dict__.keys(): # 4) for downloadable datasets in `torchtext.datasets`...
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario')
#         raw_train, raw_test, args = fetch_torchtext_dataset(args=args, dataset_name=cfg.name, root=cfg.data_path, seq_len=cfg.seq_len, tokenizer=tokenizer) 
        
#     elif cfg.name == 'TinyImageNet': # 5) for other public datasets...
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario')
#         transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
#         raw_train, raw_test, args = fetch_tinyimagenet(args=args, root=cfg.data_path, transforms=transforms)
        
#     elif cfg.name == 'CINIC10':
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario')
#         transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
#         raw_train, raw_test, args = fetch_cinic10(args=args, root=cfg.data_path, transforms=transforms)
        
#     elif 'BeerReviews' in cfg.name:
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario')
#         aspect_type = {'A': 'aroma', 'L': 'look'}
#         parsed_type = cfg.name[-1]
#         if parsed_type in ['A', 'L']:
#             aspect = aspect_type[parsed_type]
#         else:
#             err = '[DATA LOAD] Please check dataset name!'
#             logger.exception(err)
#             raise Exception(err)
#         raw_train, raw_test, args = fetch_beerreviews(args=args, root=cfg.data_path, aspect=aspect, tokenizer=tokenizer)  
        
#     elif cfg.name == 'Heart':
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario', False)
#         _check_and_raise_error(cfg.eval_type, 'local', 'evaluation type', False)
#         split_map, client_datasets, args = fetch_heart(args=args, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)
    
#     elif cfg.name == 'Adult':
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario', False)
#         _check_and_raise_error(cfg.eval_type, 'local', 'evaluation type', False)
#         split_map, client_datasets, args = fetch_adult(args=args, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)
    
#     elif cfg.name == 'Cover':
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario', False)
#         _check_and_raise_error(cfg.eval_type, 'local', 'evaluation type', False)
#         split_map, client_datasets, args = fetch_cover(args=args, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction)  
    
#     elif cfg.name == 'GLEAM':
#         _check_and_raise_error(cfg.split_type, 'pre', 'split scenario', False)
#         _check_and_raise_error(cfg.eval_type, 'local', 'evaluation type', False)
#         split_map, client_datasets, args = fetch_gleam(args=args, root=cfg.data_path, seed=cfg.seed, test_fraction=cfg.test_fraction, seq_len=cfg.seq_len)

#     else: # x) for a dataset with no support yet or incorrectly entered...
#         err = f'[DATA LOAD] Dataset `{cfg.name}` is not supported or seems incorrectly entered... please check!'
#         logger.exception(err)
#         raise Exception(err)     
#     logger.info(f'[DATA LOAD] ...successfully fetched dataset!')
    
#     ############
#     # finalize #
#     ############
#     # adjust the number of classes in binary case
#     if cfg.num_classes == 2:
#         cfg.num_classes = 1
#         cfg.criterion = 'BCEWithLogitsLoss'
        
#     # check if global holdout set is required or not
#     if cfg.eval_type == 'local':
#         raw_test = None
#     else:
#         if raw_test is None:
#             err = f'[DATA LOAD] Dataset `{cfg.name.upper()}` does not support pre-defined validation/test set, which can be used for `global` evluation... please check! (current `eval_type`=`{cfg.eval_type}`)'
#             logger.exception(err)
#             raise AssertionError(err)
            
#     # get split indices if None
#     if split_map is None:
#         logger.info(f'[DATA_SPLIT] DATA_SPLIT dataset split (split scenario: `{cfg.split_type.upper()}`)!')
#         split_map = get_data_split(args, raw_train)
#         logger.info(f'[DATA_SPLIT] ...done simulating dataset split (split scenario: `{cfg.split_type.upper()}`)!')
    
#     # construct client datasets if None
#     if client_datasets is None:
#         logger.info(f'[DATA_SPLIT] Create client datasets!')
#         client_datasets = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=min(cfg.K, os.cpu_count() - 1)) as workhorse:
#             for idx, sample_indices in TqdmToLogger(
#                 enumerate(split_map.values()), 
#                 logger=logger, 
#                 desc=f'[DATA_SPLIT] ...creating client datasets... ',
#                 total=len(split_map)
#                 ):
#                 client_datasets.append(workhorse.submit(_construct_dataset, raw_train, idx, sample_indices).result()) 
#         logger.info(f'[DATA_SPLIT] ...successfully created client datasets!')
#     gc.collect()
#     return raw_test, client_datasets    
