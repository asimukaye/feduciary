from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic

# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/09-44-15_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/14-12-34_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/15-40-06_'

# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_debug_CIFAR10/11-12-43_'
DATA_PATH ='/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_varagg_CIFAR10/13-15-52_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_varagg_CIFAR10/14-23-18_'
DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-30_varagg_CIFAR10/14-21-11_'

# DATA_PATH ='/home/asim.ukaye/fed_learning/feduciary/outputs/2023-11-22_varagg_CIFAR10/13-15-29_'

DATA_PATH ='/home/asim.ukaye/fed_learning/feduciary/outputs/2023-11-23_varagg_CIFAR10/12-20-10_'

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
    plt.close()



def plot_3d_param_histograms(in_dict: dict, tag: str, ylabel='Clients', path=f'{DATA_PATH}/histograms'):

    layers = list(in_dict.keys())
    for val in in_dict.values():
        assert isinstance(val, dict)
        feature_list = list(val.keys())

    n_cols = 4
    feat_len = len(feature_list)
    n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'projection': '3d'}, figsize=(n_cols*5, n_rows*4))

    n_bins = 100
    spacing = 10
    for i, feature in enumerate(feature_list):
        # z = np.arange(len(layers), )
        for k, layer in enumerate(layers):
            n_bins = np.minimum(n_bins, np.prod(in_dict[layer][feature].shape))
            ys, bins = np.histogram(in_dict[layer][feature], n_bins)
            xs = (bins[1:] + bins[:-1])/2
            dx = bins[1:] - bins[:-1]
            axs1[i//n_cols, i%n_cols].bar(xs, ys, zs=spacing*k, zdir='y', width=dx, alpha=0.8,   label=str(layer))

        axs1[i//n_cols, i%n_cols].set_title(feature)
        axs1[i//n_cols, i%n_cols].set_xlabel('X')
        axs1[i//n_cols, i%n_cols].set_yticks(np.arange(0, spacing*len(layers), spacing), layers)

        axs1[i//n_cols, i%n_cols].set_ylabel(ylabel)
        axs1[i//n_cols, i%n_cols].set_zlabel('Count')
        # axs1[i//n_cols, i%n_cols].legend(loc='upper right')

    handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    fig1.tight_layout()
    # fig1.legend().set_in_layout(False)
    fig1.legend(handles, labels, loc='upper center')

    if not os.path.exists(path):
        os.makedirs(path)
    fig1.savefig(f'{path}/{tag}.png')
    plt.close()
    

def plot_single_param_histograms(feature_dict: dict, path:str, tag:str):
    feature_list = list(feature_dict.keys())
    n_bins = 100

    n_cols = 4
    feat_len = len(feature_list)
    n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*4))

    for i, feature in enumerate(feature_list):
        n_bins = np.minimum(n_bins, np.prod(feature_dict[feature].shape))
        ys, bins = np.histogram(feature_dict[feature], n_bins, density=True)
        xs = (bins[1:] + bins[:-1])/2
        dx = bins[1:] - bins[:-1]
        axs1[i//n_cols, i%n_cols].bar(xs, ys, width=dx)
    
    handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    fig1.tight_layout()
    fig1.legend(handles, labels, loc='upper center')
    assert os.path.exists(path), f'Path {path} does not exist.'
    fig1.savefig(f'{path}/{tag}.png')
    plt.close()


def plot_single_histograms_wrapper(master_dict: dict, path: str, tag:str):
    for key, val in master_dict.items():
        assert isinstance(val, dict)
        path_update = f'{path}/{key}'
        if not os.path.exists(path_update):
            os.makedirs(path_update)
        plot_single_param_histograms(val, path_update, tag)

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
    del_sigmas = {}
    deltas = {}
    params = {}
    sigmas = {}


    # for file in sorted(os.listdir(f'{DATA_PATH}/varagg_debug')):
    #     np_obj = np.load(f'{DATA_PATH}/varagg_debug/{file}', allow_pickle=True)
    #     ic(file)
    #     # ic(np_obj.keys())
    #     round = np_obj['round'].item()
    #     rounds.append(round)
    #     del_sigmas[round] = np_obj['clients_del_sigma'].item()
    #     params[round] = np_obj['clients_mu']
    #     deltas[round] = np_obj['clients_delta']
    #     sigmas[round] = np_obj['clients_std']

    #     # del_sigmas[round] = del_sigma.item()
    #     # clients = list(del_sigma.item().keys())

    #     plot_single_histograms_wrapper(del_sigmas[round], f'{DATA_PATH}/histograms/del_sigma/', f'del_sigma_round_{round}')
    #     plot_3d_param_histograms(del_sigmas[round], f'del_sigma_round_{round}_3d')

    # swapped_del_sigmas = rearrange_dict_per_client(del_sigmas)
    # ic(swapped_del_sigmas.keys())
    # clients = list(swapped_del_sigmas.keys())

    # for clnt in clients:
    #     plot_3d_param_histograms(swapped_del_sigmas[clnt], f'del_sigma_client_{clnt}_3d', path=f'{DATA_PATH}/histograms/del_sigma/{clnt}', ylabel='Rounds')

    # dict_list = [del_sigmas, deltas, params, sigmas]
    # str_list = ['del_sigma', 'delta', 'mu', 'std']

    # for file in sorted(os.listdir(f'{DATA_PATH}/varagg_debug')):
    #     np_obj = np.load(f'{DATA_PATH}/varagg_debug/{file}', allow_pickle=True)
    #     ic(file)
    #     # ic(np_obj.keys())
    #     round = np_obj['round'].item()
    #     rounds.append(round)
    #     for dict_to_use, label in zip(dict_list, str_list):

    #         dict_to_use[round] = np_obj[f'clients_{label}'].item()

    #         plot_single_histograms_wrapper(dict_to_use[round], f'{DATA_PATH}/histograms/{label}', f'{label}_round_{round}')

    #         plot_3d_param_histograms(dict_to_use[round], f'{label}_round_{round}_3d', path=f'{DATA_PATH}/histograms/{label}')

    # for dict_to_use, label in zip(dict_list, str_list):
    #     swapped_dict = rearrange_dict_per_client(dict_to_use)
    #     ic(swapped_dict.keys())
    #     clients = list(swapped_dict.keys())

    #     for clnt in clients:
    #         plot_3d_param_histograms(swapped_dict[clnt], f'{label}_client_{clnt}_3d', path=f'{DATA_PATH}/histograms/{label}/{clnt}', ylabel='Rounds')


    # exit(0)
           

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
