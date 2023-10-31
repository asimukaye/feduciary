import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from icecream import ic
from collections import defaultdict

# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/09-44-15_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/14-12-34_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/15-40-06_'

# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_debug_CIFAR10/11-12-43_'
# DATA_PATH ='/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_varagg_CIFAR10/13-15-52_'
DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_varagg_CIFAR10/14-23-18_'

def plot_list(df: pd.DataFrame, list_to_plot: list, feature_list: list, tag: str):
    n_cols = 4
    feat_len = len(feature_list)
    n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, figsize=(n_cols*5, n_rows*4))
    for i, feature in enumerate(feature_list):
        for header in list_to_plot:
            axs1[i//n_cols, i%n_cols].plot(df[f'{header}.{feature}'], label=header)
        axs1[i//n_cols, i%n_cols].set_title(feature)
        # axs1[i//n_cols, i%n_cols].legend(loc='upper right')

    handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    fig1.tight_layout()
    # fig1.legend().set_in_layout(False)
    fig1.legend(handles, labels, loc='upper center')

    if not os.path.exists(f'{DATA_PATH}/plots'):
        os.makedirs(f'{DATA_PATH}/plots')
    fig1.savefig(f'{DATA_PATH}/plots/{tag}_plots.png')


def plot_3d_param_histograms(in_dict: dict, tag: str):

    layers = list(in_dict.keys())
    # feature_list = in_dict.values()
    for val in in_dict.values():
        assert isinstance(val, dict)
        feature_list = list(val.keys())

    
    ic(feature_list)
    ic (layers)
    bins = 100

    n_cols = 4
    feat_len = len(feature_list)
    n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'projection': '3d'}, figsize=(n_cols*5, n_rows*4))


    for i, feature in enumerate(feature_list):
        # z = np.arange(len(layers), )
        for k, layer in enumerate(layers):
            ys, xs = np.histogram(in_dict[layer][feature], bins, density=True )
            axs1[i//n_cols, i%n_cols].bar(xs[:-1], ys, zs=10*k, zdir='y', alpha=0.8, label=layer)

        axs1[i//n_cols, i%n_cols].set_title(feature)
        axs1[i//n_cols, i%n_cols].set_xlabel('X')
        axs1[i//n_cols, i%n_cols].set_yticks(np.arange(len(layers), 10), layers)

        axs1[i//n_cols, i%n_cols].set_ylabel('Clients')
        axs1[i//n_cols, i%n_cols].set_zlabel('Count')
        # axs1[i//n_cols, i%n_cols].legend(loc='upper right')


    handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    fig1.tight_layout()
    # fig1.legend().set_in_layout(False)
    fig1.legend(handles, labels, loc='upper center')

    if not os.path.exists(f'{DATA_PATH}/histograms'):
        os.makedirs(f'{DATA_PATH}/histograms')
    fig1.savefig(f'{DATA_PATH}/histograms/{tag}.png')
    

def plot_single_param_histograms(feature_dict: dict, path:str, tag:str):
    feature_list = list(feature_dict.keys())
    n_bins = 100

    n_cols = 4
    feat_len = len(feature_list)
    n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*4))

    for i, feature in enumerate(feature_list):
        ys, xs = np.histogram(feature_dict[feature], n_bins, density=True)
        axs1[i//n_cols, i%n_cols].bar(xs[:-1], ys)
    
    handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    fig1.tight_layout()
    fig1.legend(handles, labels, loc='upper center')
    assert os.path.exists(path), f'Path {path} does not exist.'
    fig1.savefig(f'{path}/{tag}.png')

def plot_single_histograms_wrapper(master_dict: dict, tag:str):
    for key, val in master_dict.items():
        assert isinstance(val, dict)
        path = f'{DATA_PATH}/histograms/{key}'
        if not os.path.exists(path):
            os.makedirs(path)
        plot_single_param_histograms(val, path, tag)

def rearrange_dict_per_client(main_dict: dict[int, dict[str, np.ndarray]]):
    out_dict = defaultdict(dict)   
    for round, in_dict in main_dict.items():
        for clnt, feat_dict in in_dict.items():
            out_dict[clnt][round] = feat_dict
    return out_dict

            
if __name__=='__main__':
    
    filename = f'{DATA_PATH}/varag_results.csv'

    df = pd.read_csv(filename)

    rounds = []
    del_sigma_per_client = {}
    del_sigma_per_round = {}
    del_sigmas = {}

    for file in sorted(os.listdir(f'{DATA_PATH}/varagg_debug')):
        np_obj = np.load(f'{DATA_PATH}/varagg_debug/{file}', allow_pickle=True)
        ic(file)
        # ic(np_obj.keys())
        round = np_obj['round'].item()
        rounds.append(round)
        del_sigma = np_obj['clients_del_sigma']
        del_sigmas[round] = del_sigma.item()
        clients = list(del_sigma.item().keys())

        plot_single_histograms_wrapper(del_sigmas[round], f'del_sigma_round_{round}')
        
        # plot_3d_param_histograms(del_sigma_per_client, f'del_sigma_client_{round}')

    swapped_del_sigmas = rearrange_dict_per_client(del_sigmas)
    ic(swapped_del_sigmas.keys())
    # ic(swapped_dict.values())
    del_sigma_per_round = del_sigmas[round]
    plot_3d_param_histograms(del_sigma_per_round, f'del_sigma_round_{round}_3d')

    exit(0)
           

    feature_list = []
    client_list = []
    for val in df.keys().str.split('.'):
        if val[0] == 'round':
            continue
        if val[0] == 'clients':
            client = val[1]
        if client not in client_list:
            client_list.append(client)
        feature = '.'.join(val[-3:])
        if feature not in feature_list:
            feature_list.append(feature)

    param_list = ['server_params']
    param_std_list = []
    imp_coeff_list = []
    std_weights_list = []

    for clnt in client_list:
        param_list.append(f'clients.{clnt}.param')
        param_std_list.append(f'clients.{clnt}.param_std')
        imp_coeff_list.append(f'imp_coeffs.{clnt}')
        std_weights_list.append(f'clients.{clnt}.std_weights')

    plot_list(df, param_list, feature_list, tag='param')
    plot_list(df, param_std_list, feature_list, tag='param_std')
    plot_list(df, imp_coeff_list, feature_list, tag='imp_coeffs')
    plot_list(df, std_weights_list, feature_list, tag='std_weights')
