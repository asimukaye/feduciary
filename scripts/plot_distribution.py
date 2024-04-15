from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from feduciary.data import load_raw_dataset
from feduciary.split import get_client_datasets
from feduciary.config import DatasetConfig, SplitConfig, NoiseConfig, TransformsConfig
from feduciary.common.utils import get_time
DATA_ROOT = '/home/asim.ukaye/fed_learning/feduciary/data/'

OUT_DIR = 'outputs/plots'
os.makedirs(OUT_DIR, exist_ok=True)
# DATA_PATH='/home/asim.ukaye/fed_learning/feduciary/outputs/2023-12-04_varagg_CIFAR10/21-10-54_'


def create_client_dict(client_targets):
    clients = np.array([])
    labels = np.array([])
    for i, l in enumerate(client_targets):
        clients = np.append(clients, np.full(len(l), f'cl_{i}'))
        labels = np.append(labels, np.array(l))
    return {'clients': clients, 'labels': labels}

def histogram_plots(target_dict, title):
    df = pd.DataFrame(target_dict)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.histplot(data=df, y='clients', hue='labels', multiple="stack", palette="viridis", shrink=0.6, ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title(title)
    fig.savefig(f'{OUT_DIR}/{title}.png')

   
if __name__=='__main__':
    dirichlet_alpha = 1.0
    num_class_per_client = 3
    num_splits = 6
    split_type = 'dirichlet'
    # split_type = 'patho'
    # split_type = 'iid'


    data_cfg = DatasetConfig(name='CIFAR10', data_path=DATA_ROOT,
            dataset_family='torchvision',
            transforms=TransformsConfig(),
            subsample=False,
            subsample_fraction=1.0,
            federated=True,
            test_fraction=0.2,
            seed=42,
            split_conf=SplitConfig(
                split_type=split_type,
                num_splits = num_splits,
                num_patho_clients=3,
                num_class_per_client=num_class_per_client,
                dirichlet_alpha=dirichlet_alpha,
                noise=NoiseConfig()
            )
        )
    train, test, _ = load_raw_dataset(data_cfg)

    for dirichlet_alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    # for num_class_per_client in [1, 3, 5, 7]:
    # for once in [1]:

        
        split_cfg = SplitConfig(
                split_type=split_type,
                num_splits = num_splits,
                num_patho_clients=3,
                num_class_per_client=num_class_per_client,
                dirichlet_alpha=dirichlet_alpha,
                noise=NoiseConfig())
        client_datasets = get_client_datasets(split_cfg, train, test)
        clients = np.array([])
        labels = np.array([])

        for i, c in enumerate(client_datasets):
            indices = np.array(c[0].indices)
            targets = np.array(c[0].dataset.targets)
            targets = targets[indices]
            ic(np.unique(targets, return_counts=True))
            clients = np.append(clients, np.full(targets.shape, f'cl_{i}'))
            labels = np.append(labels, targets)
        

        if split_type == 'dirichlet':
            title = f'dirichlet_alpha_{dirichlet_alpha}'
        elif split_type == 'iid':
            title = 'iid'
        elif split_type == 'patho':
            title = f'patho_{num_class_per_client}'
        elif split_type == 'one_imbalanced_client':
            title = 'one_imbalanced_client'
        
        histogram_plots({'clients': clients, 'labels': labels}, title=title)

#    weight_plots()
#    imp_coeff_comparison()