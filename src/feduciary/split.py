import logging
import numpy as np
from typing import Sequence, Protocol
import random
from torch.utils.data import Subset, Dataset
import torch
import torchvision.transforms as tvt
from feduciary.common.utils  import log_tqdm
from feduciary.config import SplitConfig
import feduciary.common.typing as fed_t
logger = logging.getLogger(__name__)

class MappedDataset(Protocol):
    @property
    def targets(self)->list:
        ...
    @property
    def class_to_idx(self)->dict:
        ...

def extract_root_dataset(subset: Subset) -> Dataset:
    if isinstance(subset.dataset, Subset):
        return extract_root_dataset(subset.dataset)
    else:
        assert isinstance(subset.dataset, Dataset), 'Unknown subset nesting' 
        return subset.dataset

def check_for_mapping(dataset: Dataset) -> MappedDataset:
    if not hasattr(dataset, 'class_to_idx'):
        raise TypeError(f'Dataset {dataset} does not have class_to_idx')
    if not hasattr(dataset, 'targets'):
        raise TypeError(f'Dataset {dataset} does not have targets')
    return dataset #type: ignore

def extract_root_dataset_and_indices(subset: Subset, indices = None) -> tuple[Dataset, np.ndarray] :
    # ic(type(subset.indices))
    if indices is None:
        indices = subset.indices
    np_indices = np.array(indices)
    # ic(np_indices)
    if isinstance(subset.dataset, Subset):
        mapped_indices = np.array(subset.dataset.indices)[np_indices]
        # ic(mapped_indices)
        return extract_root_dataset_and_indices(subset.dataset, mapped_indices)
    else:
        assert isinstance(subset.dataset, Dataset), 'Unknown subset nesting' 
        # mapped_indices = np.array(subset.indices)[in_indices]
        # ic(np_indices)
        return subset.dataset, np_indices

    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class NoisySubset(Subset):
    """Wrapper of `torch.utils.Subset` module for applying individual transform.
    """
    def __init__(self, subset: Subset,  mean:float, std: float):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self._subset = subset
        self.noise = AddGaussianNoise(mean, std)

    def __getitem__(self, idx):
        inputs, targets = self._subset[idx]
        return self.noise(inputs), targets

    def __len__(self):
        return len(self.indices)

    
    def __repr__(self):
        return f'{repr(self.dataset)}_GaussianNoise'

class LabelFlippedSubset(Subset):
    """Wrapper of `torch.utils.Subset` module for label flipping.
    """
    def __init__(self, subset: Subset,  flip_pct: float):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.subset = self._flip_set(subset, flip_pct) 

    def _flip_set(self, subset: Subset, flip_pct: float):
        total_size = len(subset)
        dataset, mapped_ids = extract_root_dataset_and_indices(subset)
        dataset = check_for_mapping(dataset)

        # ic(total_size, len(mapped_ids))

        samples = np.random.choice(total_size, size=int(flip_pct*total_size), replace=False)
        

        selected_indices = mapped_ids[samples]
        # ic(samples, selected_indices)
        class_ids = list(dataset.class_to_idx.values())
        for idx, dataset_idx in zip(samples, selected_indices):
            _, lbl = subset[idx]
            assert lbl == dataset.targets[dataset_idx]
            # ic(lbl, )
            excluded_labels = [cid for cid in class_ids if cid != lbl]
            # changed_label = np.random.choice(excluded_labels)
            # ic(changed_label)
            dataset.targets[dataset_idx] = np.random.choice(excluded_labels)
            # print('\n')
        return subset
    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset)}_LabelFlipped'

    
def get_iid_split(dataset: Subset, num_splits: int, seed: int = 42) -> dict[int, np.ndarray]:
    shuffled_indices = np.random.permutation(len(dataset))
        
    # get adjusted indices
    split_indices = np.array_split(shuffled_indices, num_splits)
    
    # construct a hashmap
    split_map = {k: split_indices[k] for k in range(num_splits)}
    return split_map

def get_unbalanced_split(dataset: Subset, num_splits: int) -> dict[int, np.ndarray]:
     # shuffle sample indices
    shuffled_indices = np.random.permutation(len(dataset))
    
    # split indices by number of clients
    split_indices = np.array_split(shuffled_indices, num_splits)
        
    # randomly remove some proportion (1% ~ 5%) of data
    keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))
        
    # get adjusted indices
    split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
    
    # construct a hashmap
    split_map = {k: split_indices[k] for k in range(num_splits)}
    return split_map

def get_one_patho_client_split(dataset: Subset, num_splits) -> dict[int, np.ndarray]:
    total_size = len(dataset)
    shuffled_indices = np.random.permutation(total_size)
    # client 1 gets half the size of data compared to rest
    c1_count = int(total_size/(2*num_splits - 1))
    c1_share = shuffled_indices[:c1_count]

    rest_share = np.array_split(shuffled_indices[c1_count:], num_splits-1)

    # assert len(c1_share) + len(share) for share in rest_share == total_size
    split_map = {}
    size_check = [c1_count]
    split_map[0] = c1_share
    for k, others_share in enumerate(rest_share):
        split_map[k+1] = others_share
        size_check.append(len(others_share))
    # logger.info(f'Size total after split : {size_check}, total size: {total_size}')
    logger.info(f'Split map sizes: {size_check}')
    return split_map

def get_one_imbalanced_client_split(dataset: Subset, num_splits: int) -> dict[int, np.ndarray]:
    total_size = len(dataset)
    shuffled_indices = np.random.permutation(total_size)
    # client 1 gets half the size of data compared to rest
    c1_count = int(total_size/(2*num_splits - 1))
    c1_share = shuffled_indices[:c1_count]

    rest_share = np.array_split(shuffled_indices[c1_count:], num_splits-1)

    # assert len(c1_share) + len(share) for share in rest_share == total_size
    split_map = {}
    size_check = [c1_count]
    split_map[0] = c1_share
    for k, others_share in enumerate(rest_share):
        split_map[k+1] = others_share
        size_check.append(len(others_share))
    # logger.info(f'Size total after split : {size_check}, total size: {total_size}')
    logger.info(f'Split map sizes: {size_check}')
    return split_map
    

# TODO: Understand this function from FedAvg Paper
def get_patho_split(dataset: Subset, num_splits: int, num_classes: int, mincls: int) -> dict[int, np.ndarray]:
    try:
        assert mincls >= 2
    except AssertionError as e:
        logger.exception("[DATA_SPLIT] Each client should have samples from at least 2 distinct classes!")
        raise e
    
    # get unique class labels and their count
    inferred_classes, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
    # split the indices by class labels
    class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
    
    assert len(inferred_classes) == num_classes, 'Inferred classes do not match the expected number of classes'
    
    # divide shards
    num_shards_per_class = num_splits* mincls // num_classes
    if num_shards_per_class < 1:
        err = f'[DATA_SPLIT] Increase the number of minimum class (`args.mincls` > {mincls}) or the number of participating clients (`args.K` > {num_splits})!'
        logger.exception(err)
        raise Exception(err)
    
    # split class indices again into groups, each having the designated number of shards
    split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
    
    # make hashmap to track remaining shards to be assigned per client
    class_shards_counts = dict(zip([i for i in range(num_classes)], [len(split_idx) for split_idx in split_indices]))

    # assign divided shards to clients
    assigned_shards = []
    for _ in range(num_classes):
        # update selection proability according to the count of reamining shards
        # i.e., do NOT sample from class having no remaining shards
        selection_prob = np.where(np.array(list(class_shards_counts.values())) > 0, 1., 0.)
        selection_prob /= sum(selection_prob)
        
        # select classes to be considered
        try:
            selected_classes = np.random.choice(num_classes, mincls, replace=False, p=selection_prob)
        except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
            selected_classes = np.random.choice(num_classes, mincls, replace=True, p=selection_prob)
        
        # assign shards in randomly selected classes to current client
        for it, class_idx in enumerate(selected_classes):
            selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
            selected_shards = split_indices[class_idx].pop(selected_shard_indices)
            if it == 0:
                assigned_shards.append([selected_shards])
            else:
                assigned_shards[-1].append(selected_shards)
            class_shards_counts[class_idx] -= 1
        else:
            assigned_shards[-1] = np.concatenate(assigned_shards[-1])

    # construct a hashmap
    split_map = {k: assigned_shards[k] for k in range(num_splits)}
    return split_map


def sample_with_mask(mask, ideal_samples_counts, concentration, num_classes, need_adjustment=False):
    num_remaining_classes = int(mask.sum())
    
    # sample class selection probabilities based on Dirichlet distribution with concentration parameter (`diri_alpha`)
    selection_prob_raw = np.random.dirichlet(alpha=np.ones(num_remaining_classes) * concentration, size=1).squeeze()
    selection_prob = mask.copy()
    selection_prob[selection_prob == 1.] = selection_prob_raw
    selection_prob /= selection_prob.sum()

    # calculate per-class sample counts based on selection probabilities
    if need_adjustment: # if remaining samples are not enough, force adjusting sample sizes...
        selected_counts = (selection_prob * ideal_samples_counts * np.random.uniform(low=0.0, high=1.0, size=len(selection_prob))).astype(int)
    else:
        selected_counts = (selection_prob * ideal_samples_counts).astype(int)
    return selected_counts


# TODO: understand this split from paper
def get_dirichlet_split_2(dataset: Dataset, num_splits, num_classes, cncntrtn)-> dict[int, np.ndarray]:
           
    # get indices by class labels
    _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
    class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
    
    # make hashmap to track remaining samples per class
    class_samples_counts = dict(zip([i for i in range(
        num_classes)], [len(class_idx) for class_idx in class_indices]))
    
    # calculate ideal samples counts per client
    ideal_samples_counts = len(dataset.targets) // num_classes
    if ideal_samples_counts < 1:
        err = f'[DATA_SPLIT] Decrease the number of participating clients (`args.K` < {num_splits})!'
        logger.exception(err)
        raise Exception(err)

    # assign divided shards to clients
    assigned_indices = []
    for k in log_tqdm(
        range(num_splits), 
        logger=logger,
        desc='[DATA_SPLIT] assigning to clients '
        ):
        # update mask according to the count of reamining samples per class
        # i.e., do NOT sample from class having no remaining samples
        remaining_mask = np.where(np.array(list(class_samples_counts.values())) > 0, 1., 0.)
        selected_counts = sample_with_mask(remaining_mask, ideal_samples_counts, cncntrtn, num_classes)

        # check if enough samples exist per selected class
        expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)
        valid_mask = np.where(expected_counts < 0, 1., 0.)
        
        # if not, resample until enough samples are secured
        while sum(valid_mask) > 0:
            # resample from other classes instead of currently selected ones
            adjusted_mask = (remaining_mask.astype(bool) & (~valid_mask.astype(bool))).astype(float)
            
            # calculate again if enoush samples exist or not
            selected_counts = sample_with_mask(adjusted_mask, ideal_samples_counts, cncntrtn, num_classes, need_adjustment=True)    
            expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)

            # update mask for checking a termniation condition
            valid_mask = np.where(expected_counts < 0, 1., 0.)
            
        # assign shards in randomly selected classes to current client
        indices = []
        for it, counts in enumerate(selected_counts):
            # get indices from the selected class
            selected_indices = class_indices[it][:counts]
            indices.extend(selected_indices)
            
            # update indices and statistics
            class_indices[it] = class_indices[it][counts:]
            class_samples_counts[it] -= counts
        else:
            assigned_indices.append(indices)

    # construct a hashmap
    split_map = {k: assigned_indices[k] for k in range(num_splits)}
    return split_map


def dirichlet_noniid_split(dataset: Subset, n_clients: int, alpha: float,) -> dict[int, np.ndarray]:
    """Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha.
    Args:
        train_labels: ndarray of train_labels.
        alpha: the parameter of Dirichlet distribution.
        n_clients: number of clients.
    Returns:
        client_idcs: a list containing sample idcs of clients.
    """
    target_set = Subset(dataset.dataset.targets, dataset.indices)
    train_labels = np.array(target_set)
    # train_labels = np.array(train_labels)
    n_classes = train_labels.max()+1
    # (n_classes, n_clients), label distribution matrix, indicating the
    # proportion of each label's data divided into each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    # (n_classes, ...), indicating the sample indices for each label
    class_idcs = [np.argwhere(train_labels == y).flatten()
                    for y in range(n_classes)]

    # Indicates the sample indices of each client
    client_idcs = [[] for _ in range(n_clients)]
    for c_idcs, fracs in zip(class_idcs, label_distribution):
        # `np.split` divides the sample indices of each class, i.e.`c_idcs`
        # into `n_clients` subsets according to the proportion `fracs`.
        # `i` indicates the i-th client, `idcs` indicates its sample indices
        for i, idcs in enumerate(np.split(c_idcs, (
                np.cumsum(fracs)[:-1] * len(c_idcs))
                .astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = {k: np.concatenate(idcs) for k, idcs in zip(range(n_clients),client_idcs)}

    return client_idcs


def pathological_non_iid_split(dataset: Subset, n_clients: int,  n_classes_per_client: int) -> dict[int, np.ndarray]:
    target_set = Subset(dataset.dataset.targets, dataset.indices)
    train_labels = np.array(target_set)
    # ic(train_labels.shape)
    # ic(train_labels[:10])
    n_classes = train_labels.max()+1
    data_idcs = list(range(len(train_labels)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        label = train_labels[idx]
        label2index[label].append(idx)

    # ic(label2index)
    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(lst, g):
        """Divides the list `l` into `g` i.i.d. groups, i.e.direct splitting.
        Each group has `int(len(l)/g)` or `int(len(l)/g)+1` elements.
        Returns a list of different groups.
        """
        num_elems = len(lst)
        group_size = int(len(lst) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(lst[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(lst[bi + group_size * i:bi + group_size * (i + 1)])
        return glist

    n_shards = n_clients * n_classes_per_client
    # Divide the sample indices into `n_shards` i.i.d. shards
    shards = iid_divide(sorted_idcs, n_shards)

    np.random.shuffle(shards)
    # Then split the shards into `n_client` parts
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # Here `shard` is the sample indices of a shard (a list)
            # `+= shard` is to merge the list `shard` into the list
            # `clients_idcs[client_id]`
            clients_idcs[client_id] += shard

    clients_idcs = {k: np.array(idcs) for k, idcs in zip(range(n_clients),clients_idcs)}
    return clients_idcs

def get_split_map(cfg: SplitConfig, dataset: Subset) -> dict[int, np.ndarray]:
    """Split data indices using labels.
    Args:
        cfg (DatasetConfig): Master dataset configuration class
        dataset (dataset): raw dataset instance to be split 
        
    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
    """
    match cfg.split_type:
        case 'iid' | 'one_noisy_client' | 'one_label_flipped_client'|'n_label_flipped_clients'| 'n_noisy_clients':
            split_map = get_iid_split(dataset, cfg.num_splits)
            return split_map

        case 'unbalanced':
            split_map = get_unbalanced_split(dataset, cfg.num_splits)
            return split_map

        case 'one_imbalanced_client':
            split_map = get_one_imbalanced_client_split(dataset, cfg.num_splits)
            return split_map

        case 'patho':
            split_map = pathological_non_iid_split(dataset, cfg.num_splits, cfg.num_class_per_client)  
            return split_map
        #     raise NotImplementedError
        case 'dirichlet':
            split_map = dirichlet_noniid_split(dataset, cfg.num_splits, cfg.dirichlet_alpha)
            return split_map
        case 'leaf' |'fedvis'|'flamby':
            logger.info('[DATA_SPLIT] Using pre-defined split.')
            raise NotImplementedError
        case _ :
            logger.error('[DATA_SPLIT] Unknown datasplit type')
            raise NotImplementedError



def _construct_client_dataset(raw_train: Dataset, raw_test: Dataset, train_indices, test_indices) ->tuple[Subset, Subset]:
    train_set = Subset(raw_train, train_indices)
    test_set = Subset(raw_test, test_indices)
    return (train_set, test_set)


def get_client_datasets(cfg: SplitConfig, train_dataset: Subset, test_dataset, match_train_distribution=False) -> list[fed_t.DatasetPair_t] :
    # logger.info(f'[DATA_SPLIT] dataset split: `{cfg.split_type.upper()}`')   
    split_map = get_split_map(cfg, train_dataset)
    if match_train_distribution:
        test_split_map = get_split_map(cfg, test_dataset)
    else:
        test_split_map = get_iid_split(test_dataset, cfg.num_splits)

    assert len(split_map) == len(test_split_map), 'Train and test split maps should be of same length'
    logger.info(f'[DATA_SPLIT] Simulated dataset split : `{cfg.split_type.upper()}`')
    
    # construct client datasets if None
    cfg.test_fractions = []
    client_datasets = []
    for idx, train_indices in enumerate(split_map.values()):
        train_set, test_set = _construct_client_dataset(train_dataset, test_dataset, train_indices, test_indices=test_split_map[idx])
        cfg.test_fractions.append(len(test_set)/len(train_set))
        client_datasets.append((train_set, test_set))
    
    match cfg.split_type:
        case 'one_noisy_client':
            train, test = client_datasets[0]
            patho_train = NoisySubset(train, cfg.noise.mu, cfg.noise.sigma)
            if match_train_distribution:
                test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
            client_datasets[0] = patho_train, test
        case 'one_label_flipped_client':
            train, test = client_datasets[0]
            patho_train = LabelFlippedSubset(train, cfg.noise.flip_percent)
            if match_train_distribution:
                test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
            client_datasets[0] = patho_train, test
        case 'n_label_flipped_clients':
            for idx in range(cfg.num_patho_clients):
                train, test = client_datasets[idx]
                patho_train = LabelFlippedSubset(train, cfg.noise.flip_percent)
                if match_train_distribution:
                    test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
                client_datasets[idx] = patho_train, test
        case 'n_noisy_clients':
            for idx in range(cfg.num_patho_clients):
                train, test = client_datasets[idx]
                patho_train = NoisySubset(train, cfg.noise.mu, cfg.noise.sigma)
                if match_train_distribution:
                    test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
                client_datasets[idx] = patho_train, test
        case _:
            pass
    logger.debug(f'[DATA_SPLIT] Created client datasets!')
    logger.debug(f'[DATA_SPLIT] Split fractions: {cfg.test_fractions}')



    return client_datasets


# class SubsetWrapper(Subset):
#     # NOTE: looks like this is just a 
#     """Wrapper of `torch.utils.Subset` module for applying individual transform.
#     """
#     def __init__(self, subset:Subset,  suffix:str):
#         self.subset = subset
#         self.suffix = suffix

#     def __getitem__(self, index):
#         inputs, targets = self.subset[index]
#         return inputs, targets

#     def __len__(self):
#         return len(self.subset)
    
#     def __repr__(self):
#         return f'{repr(self.subset.dataset)} {self.suffix}'



# def get_data_split(args:DatasetConfig, dataset: Dataset):
#     """Split data indices using labels.
    
#     Args:
#         args (argparser): arguments
#         dataset (dataset): raw dataset instance to be split 
        
#     Returns:
#         split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
#     """
#     # IID split (i.e., statistical homogeneity)
#     if args.split_type == 'iid': 
#         # shuffle sample indices
#         shuffled_indices = np.random.permutation(len(dataset))
        
#         # get adjusted indices
#         split_indices = np.array_split(shuffled_indices, args.K)
        
#         # construct a hashmap
#         split_map = {k: split_indices[k] for k in range(args.K)}
#         return split_map
    
#     # non-IID split by sample unbalancedness
#     if args.split_type == 'unbalanced': 
#         # shuffle sample indices
#         shuffled_indices = np.random.permutation(len(dataset))
        
#         # split indices by number of clients
#         split_indices = np.array_split(shuffled_indices, args.K)
            
#         # randomly remove some proportion (1% ~ 5%) of data
#         keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))
            
#         # get adjusted indices
#         split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
        
#         # construct a hashmap
#         split_map = {k: split_indices[k] for k in range(args.K)}
#         return split_map
    
#     # Non-IID split proposed in (McMahan et al., 2016); each client has samples from at least two different classes
#     elif args.split_type == 'patho': 
#         try:
#             assert args.mincls >= 2
#         except AssertionError as e:
#             logger.exception("[DATA_SPLIT] Each client should have samples from at least 2 distinct classes!")
#             raise e
        
#         # get indices by class labels
#         _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
#         class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
            
#         # divide shards
#         num_shards_per_class = args.K * args.mincls // args.num_classes
#         if num_shards_per_class < 1:
#             err = f'[DATA_SPLIT] Increase the number of minimum class (`args.mincls` > {args.mincls}) or the number of participating clients (`args.K` > {args.K})!'
#             logger.exception(err)
#             raise Exception(err)
        
#         # split class indices again into groups, each having the designated number of shards
#         split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
        
#         # make hashmap to track remaining shards to be assigned per client
#         class_shards_counts = dict(zip([i for i in range(args.num_classes)], [len(split_idx) for split_idx in split_indices]))

#         # assign divided shards to clients
#         assigned_shards = []
#         for _ in log_tqdm(
#             range(args.K), 
#             logger=logger,
#             desc='[DATA_SPLIT] ...assigning to clients... '
#             ):
#             # update selection proability according to the count of reamining shards
#             # i.e., do NOT sample from class having no remaining shards
#             selection_prob = np.where(np.array(list(class_shards_counts.values())) > 0, 1., 0.)
#             selection_prob /= sum(selection_prob)
            
#             # select classes to be considered
#             try:
#                 selected_classes = np.random.choice(args.num_classes, args.mincls, replace=False, p=selection_prob)
#             except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
#                 selected_classes = np.random.choice(args.num_classes, args.mincls, replace=True, p=selection_prob)
            
#             # assign shards in randomly selected classes to current client
#             for it, class_idx in enumerate(selected_classes):
#                 selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
#                 selected_shards = split_indices[class_idx].pop(selected_shard_indices)
#                 if it == 0:
#                     assigned_shards.append([selected_shards])
#                 else:
#                     assigned_shards[-1].append(selected_shards)
#                 class_shards_counts[class_idx] -= 1
#             else:
#                 assigned_shards[-1] = np.concatenate(assigned_shards[-1])

#         # construct a hashmap
#         split_map = {k: assigned_shards[k] for k in range(args.K)}
#         return split_map
    
#     # Non-IID split proposed in (Hsu et al., 2019); simulation of non-IID split scenario using Dirichlet distribution
#     elif args.split_type == 'diri':
#         def sample_with_mask(mask, ideal_samples_counts, concentration, num_classes, need_adjustment=False):
#             num_remaining_classes = int(mask.sum())
            
#             # sample class selection probabilities based on Dirichlet distribution with concentration parameter (`diri_alpha`)
#             selection_prob_raw = np.random.dirichlet(alpha=np.ones(num_remaining_classes) * concentration, size=1).squeeze()
#             selection_prob = mask.copy()
#             selection_prob[selection_prob == 1.] = selection_prob_raw
#             selection_prob /= selection_prob.sum()

#             # calculate per-class sample counts based on selection probabilities
#             if need_adjustment: # if remaining samples are not enough, force adjusting sample sizes...
#                 selected_counts = (selection_prob * ideal_samples_counts * np.random.uniform(low=0.0, high=1.0, size=len(selection_prob))).astype(int)
#             else:
#                 selected_counts = (selection_prob * ideal_samples_counts).astype(int)
#             return selected_counts
            
#         # get indices by class labels
#         _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
#         class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
        
#         # make hashmap to track remaining samples per class
#         class_samples_counts = dict(zip([i for i in range(args.num_classes)], [len(class_idx) for class_idx in class_indices]))
        
#         # calculate ideal samples counts per client
#         ideal_samples_counts = len(dataset.targets) // args.K
#         if ideal_samples_counts < 1:
#             err = f'[DATA_SPLIT] Decrease the number of participating clients (`args.K` < {args.K})!'
#             logger.exception(err)
#             raise Exception(err)

#         # assign divided shards to clients
#         assigned_indices = []
#         for k in log_tqdm(
#             range(args.K), 
#             logger=logger,
#             desc='[DATA_SPLIT] ...assigning to clients... '
#             ):
#             # update mask according to the count of reamining samples per class
#             # i.e., do NOT sample from class having no remaining samples
#             remaining_mask = np.where(np.array(list(class_samples_counts.values())) > 0, 1., 0.)
#             selected_counts = sample_with_mask(remaining_mask, ideal_samples_counts, args.cncntrtn, args.num_classes)

#             # check if enough samples exist per selected class
#             expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)
#             valid_mask = np.where(expected_counts < 0, 1., 0.)
            
#             # if not, resample until enough samples are secured
#             while sum(valid_mask) > 0:
#                 # resample from other classes instead of currently selected ones
#                 adjusted_mask = (remaining_mask.astype(bool) & (~valid_mask.astype(bool))).astype(float)
                
#                 # calculate again if enoush samples exist or not
#                 selected_counts = sample_with_mask(adjusted_mask, ideal_samples_counts, args.cncntrtn, args.num_classes, need_adjustment=True)    
#                 expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)

#                 # update mask for checking a termniation condition
#                 valid_mask = np.where(expected_counts < 0, 1., 0.)
                
#             # assign shards in randomly selected classes to current client
#             indices = []
#             for it, counts in enumerate(selected_counts):
#                 # get indices from the selected class
#                 selected_indices = class_indices[it][:counts]
#                 indices.extend(selected_indices)
                
#                 # update indices and statistics
#                 class_indices[it] = class_indices[it][counts:]
#                 class_samples_counts[it] -= counts
#             else:
#                 assigned_indices.append(indices)

#         # construct a hashmap
#         split_map = {k: assigned_indices[k] for k in range(args.K)}
#         return split_map
#     # `leaf` - LEAF benchmark (Caldas et al., 2018); `fedvis` - Federated Vision Datasets (Hsu, Qi and Brown, 2020)
#     elif args.split_type in ['leaf', 'fedvis']: 
#         logger.info('[DATA_SPLIT] Use pre-defined split!')
