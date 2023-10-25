import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from icecream import ic


DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/09-44-15_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/14-12-34_'
# DATA_PATH = '/home/asim.ukaye/fed_learning/feduciary/outputs/2023-10-24_varagg_CIFAR10/15-40-06_'

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

if __name__=='__main__':
    
    filename = f'{DATA_PATH}/varag_results.csv'

    df = pd.read_csv(filename)
    # ic(df.keys().str.split('.'))
    feature_list = []
    client_list = []
    for val in df.keys().str.split('.'):
        if val[0] == 'round':
            continue
        if val[0] == 'clients':
            client = val[1]
        if client not in client_list:
            client_list.append(client)
        # ic(val[0], val[1], val[2])
        feature = '.'.join(val[-3:])
        if feature not in feature_list:
            feature_list.append(feature)

        # feature_list = '.'.join(val[-3:])
    # ic(feature_list)
    # ic(df[f'server_params.{feature}'])
    # param = {}
    # for feature in feature_list:
    #     param[feature] = 0


    # ic(client_list)
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


    # n_cols = 4
    # feat_len = len(feature_list)
    # n_rows = feat_len//n_cols + bool(feat_len%n_cols)
    # # axs1: np.ndarray[plt.Axes]
    # fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, figsize=(n_cols*5, n_rows*4))
    # for i, feature in enumerate(feature_list):
    #     for header in param_list:
    #         axs1[i//n_cols, i%n_cols].plot(df[f'{header}.{feature}'], label=header)
    #     axs1[i//n_cols, i%n_cols].set_title(feature)
    #     # axs1[i//n_cols, i%n_cols].legend(loc='upper right')

    # handles, labels = axs1[i//n_cols, i%n_cols].get_legend_handles_labels()
    # fig1.tight_layout()
    # fig1.legend(handles, labels, loc='upper center')

    # fig1.savefig('param_plots.png')