import torch
import logging
from feduciary.config import DatasetModelSpec
from torch.utils.data import Dataset
import medmnist
import numpy as np
logger = logging.getLogger(__name__)

class MedmnistClassificationDataset(Dataset): 
    def __init__(self, dataset: medmnist.BloodMNIST, dataset_name, suffix):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.suffix = suffix
        self.data = dataset.imgs
        self.targets: np.ndarray = dataset.labels
        self.indices = np.arange(len(self.dataset))
        self.class_to_idx = {v:int(k) for k,v in dataset.info['label'].items()}

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, targets.squeeze()

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[{self.dataset_name}] {self.suffix}'

# helper method to fetch dataset from `torchvision.datasets`
    
def fetch_medmnist_dataset(dataset_name:str, root, transforms)->tuple[Dataset, Dataset, DatasetModelSpec]:
    logger.debug(f'[DATA LOAD] Fetching dataset: {dataset_name.upper()}')
    # Initialize dataset dependent model spec
    model_spec =  DatasetModelSpec(num_classes=2,in_channels=1)

    # default arguments
    ALL_DATASETS = [s.lower() for s in medmnist.__dict__.keys() if 'MNIST' in s]
    
    # print(ALL_DATASETS)
    assert dataset_name in ALL_DATASETS, f"Dataset {dataset_name} not found in medmnist"

    info = medmnist.INFO[dataset_name]
    task = info['task']
    # print(task)
    model_spec.in_channels = info['n_channels']
    model_spec.num_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    transforms[0]
    
    raw_train = DataClass(root=root, split='train', transform=transforms[0], download=True)

    raw_train = MedmnistClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')

    raw_test = DataClass(root=root, split='test', transform=transforms[1], download=True)
    raw_test = MedmnistClassificationDataset(raw_test, dataset_name.upper(), 'SERVER')
    # ic(len(raw_test))
    
    logger.info(f'[DATA LOAD] Fetched dataset: {dataset_name.upper()}')

    return raw_train, raw_test, model_spec