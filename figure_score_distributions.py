#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython import embed
from config import Config
from utils import load_hdf5_as_data_frame, smooth_histogram


def scott_bins(data):
    # Compute Scott's bin width
    n = len(data)
    std_dev = np.std(data)
    bin_width = 3.5 * std_dev / np.cbrt(n)
    data_range = data.max() - data.min()

    # Calculate number of bins
    bins = int(np.ceil(data_range / bin_width))
    return bins


def freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(len(data))
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return bins


def get_label_pos(ax, threshold, y_per, x_per):
    # Get x and y axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Vertical position: x% up the y-axis range
    y_pos = y_max * y_per

    # Horizontal offset: x% of x-axis range
    x_offset = (x_max - x_min) * x_per

    # Decide text position: shift right unless near right edge
    if threshold + x_offset > x_max:
        x_text = threshold - x_offset
        ha = 'right'
    else:
        x_text = threshold + x_offset
        ha = 'left'

    return x_text, y_pos, ha


def plot_null_score_histogram(data, threshold):
    bins = int(scott_bins(data) / 2)
    x_vals, kde_vals = smooth_histogram(data[::1000], bandwidth=0.08, res=1000)

    # Figure
    fig_width_cm = 10
    fig_height_cm = 10
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    # fig = plt.figure(figsize=(fig_width, fig_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.2, top=0.8, bottom=0.2, right=0.8, wspace=0, hspace=0)

    # Plot histogram
    # ax.hist(data, bins=bins, weights=np.ones(len(data)) / len(data))
    ax.hist(data, bins=bins, density=True, color='black', alpha=0.5)
    ax.plot(x_vals, kde_vals, color='black', lw=2, alpha=0.8)
    ax.axvline(threshold, color='red', ls=':', lw=1)
    x_pos, y_pos, ha = get_label_pos(ax, threshold, y_per=0.9, x_per=0.005)
    ax.text(x_pos, y_pos, r'$95^{\mathrm{th}}$ per.', fontsize=6, ha=ha, va='center', color='red')

    x_pos, y_pos, ha = get_label_pos(ax, threshold, y_per=0.8, x_per=0.005)
    ax.text(x_pos, y_pos, f'n={data.shape[0]}', fontsize=6, ha=ha, va='center', color='red')

    # ax.text(threshold+0.02, 32, f'{threshold:.2f}', fontsize=10, ha='left', va='center', color='red')

    # ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_xlim((-0.3, 0.3))
    ax.set_ylim((0, 40))
    ax.set_title('scores_null')

    # Store Figure to HDD
    plt.savefig(f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/histograms/scores_null.jpg', dpi=600)
    plt.close(fig)


def plot_score_histogram(data, threshold, file_name):
    rois_above_th = int((data >= threshold).sum())
    percent_above_th = rois_above_th / len(data)

    # bins = int(scott_bins(data))
    bins = 20
    print(f'{file_name}, bins={bins}')
    # x_vals, kde_vals = smooth_histogram(data, bandwidth=0.05, res=1000)

    # Figure
    fig_width_cm = 10
    fig_height_cm = 10
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    # fig = plt.figure(figsize=(fig_width, fig_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.2, top=0.8, bottom=0.2, right=0.8, wspace=0, hspace=0)

    # Plot histogram
    # ax.hist(data, bins=bins, weights=np.ones(len(data)) / len(data))
    ax.hist(data, bins=bins, density=True, color='black', alpha=0.5)
    # ax.plot(x_vals, kde_vals, color='black', lw=1, alpha=0.5)
    ax.axvline(threshold, color='red', ls=':', lw=1)
    x_pos, y_pos, ha = get_label_pos(ax, threshold, y_per=0.9, x_per=0.005)
    ax.text(x_pos, y_pos, f'sign. threshold ({threshold:.2f}) \n {percent_above_th*100:.0f} % sign. responses', fontsize=6, ha=ha, va='center', color='red')

    x_pos, y_pos, ha = get_label_pos(ax, threshold, y_per=0.8, x_per=0.005)
    ax.text(x_pos, y_pos, f'n={data.shape[0]}', fontsize=6, ha=ha, va='center', color='red')

    # ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    # ax.set_xlim((np.min(data), np.max(data)))
    ax.set_xlim((-0.3, 4))

    ax.set_title(file_name)
    # ax.set_ylim((0, 40))

    # Store Figure to HDD
    # plt.show()
    # embed()
    # exit()
    plt.savefig(f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/histograms/{file_name}.jpg', dpi=600)
    plt.close(fig)


def load_data():
    null_scores_file = f'{Config.BASE_DIR}/data/scores_null_distribution_10k_permutations.h5'
    scores_file = f'{Config.BASE_DIR}/data/scores_null_distribution_10k_permutations.h5'
    null_scores = load_hdf5_as_data_frame(null_scores_file).iloc[:, 0].to_numpy()
    linear_scores = pd.read_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv')
    return linear_scores, null_scores


def main():
    linear_scores, null_scores = load_data()
    th = np.percentile(null_scores, 90)

    # plot_null_score_histogram(null_scores, th)

    # reg_names = linear_scores['reg'].unique()
    reg_names = [
        'moving_target_MEAN',
        'moving_target_motor_MEAN',
        'grating_appears',
        'grating_appears_motor',
        'grating_0',
        'grating_0_motor',
        'grating_180',
        'grating_180_motor',
        'grating_disappears',
        'grating_disappears_motor',
        'bright_loom',
        'bright_loom_motor',
        'dark_loom',
        'dark_loom_motor',
        'bright_flash',
        'bright_flash_motor',
        'dark_flash',
        'dark_flash_motor',
        'motor_spontaneous',
        'motor_spontaneous_MEAN'
    ]

    th = 0.09
    # s1 = 'motor_spontaneous_MEAN'
    s1 = 'bright_loom'
    s2 = 'bright_loom_motor'
    s2_motor = 'grating_0_motor'

    score_s1 = linear_scores[linear_scores['reg'] == s1].reset_index(drop=True)
    score_s2 = linear_scores[linear_scores['reg'] == s2].reset_index(drop=True)
    score_s2_motor = linear_scores[linear_scores['reg'] == s2_motor].reset_index(drop=True)

    # idx = (score_s2_motor['score'] <= th) * (score_s2['score'] > th)
    # idx = (score_s2_motor['score'] <= th)
    idx = (score_s2['score'] > th) * (score_s1['score'] > th)
    score_s2_vis = score_s2[idx]
    score_s1_vis = score_s1[idx]

    # import statsmodels.api as sm
    # # Create design matrix and add Intercept
    # design_matrix = sm.add_constant(score_s2_vis['score'].to_numpy())
    # # Linear Regression
    # # Compute LM-OLS Model
    # model = sm.OLS(score_s1_vis['score'].to_numpy(), design_matrix).fit()
    # r2 = model.rsquared
    # cf = model.params[1]

    fig, ax = plt.subplots()
    ax.scatter(score_s1['score'], score_s2['score'], color='tab:blue', alpha=1)
    ax.scatter(score_s1_vis['score'], score_s2_vis['score'], color='tab:green', alpha=1)
    ax.axvline(th, color='black', ls=':')
    ax.axhline(th, color='black', ls=':')

    # plt.xlim(-1, 4)
    # plt.ylim(-1, 4)
    ax.set_xlabel(f'Score ({s1})')
    ax.set_ylabel(f'Score ({s2})')
    plt.title(f'{s1}, {s2}')
    plt.show()

    plot_score_histogram(score_s1[np.invert(idx)]['score'], th, 'test')

    exit()
    # Plot Score Histograms
    for reg_name in reg_names:
        score_data = linear_scores[linear_scores['reg'] == reg_name]['score'].to_numpy()
        plot_score_histogram(score_data, th, reg_name)


if __name__ == "__main__":
    main()
