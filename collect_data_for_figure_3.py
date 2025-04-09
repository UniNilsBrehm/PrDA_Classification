#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
from utils import load_hdf5_as_dict, get_rois_per_sweep


def load_data():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_events_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_events.h5'
    ca_labels_file = f'{base_dir}/data/df_f_data_labels.csv'
    linear_scoring_file = f'{base_dir}/data/linear_scoring_results.csv'

    ca_labels = pd.read_csv(ca_labels_file)
    linear_scoring_results = pd.read_csv(linear_scoring_file)
    vr_events = load_hdf5_as_dict(vr_events_dir)
    sw_rois = get_rois_per_sweep(ca_labels)

    return ca_labels, vr_events, sw_rois, linear_scoring_results


def get_count_spontaneous_swims(vr_events, rois_per_sweep):
    spontaneous_counts = dict()

    stimulus_counts = {
        'moving_target': 0,
        'grating_appears': 0,
        'grating_0': 0,
        'grating_180': 0,
        'grating_disappears': 0,
        'bright_loom': 0,
        'dark_loom': 0,
        'bright_flash': 0,
        'dark_flash': 0,
    }
    stimulus_counts_rois = stimulus_counts.copy()

    for sw in vr_events:
        rois_count = rois_per_sweep[sw].shape[0]
        count = (vr_events[sw]['stimulus'] == 'spontaneous').sum()
        spontaneous_counts[sw] = int(count)

        # Get swims for stimuli
        for s_type in stimulus_counts:
            if s_type.startswith('moving_target'):
                count = int((vr_events[sw]['stimulus'] == 'moving_target_01').sum() > 0 or (vr_events[sw]['stimulus'] == 'moving_target_02').sum() > 0)
            else:
                count = int((vr_events[sw]['stimulus'] == s_type).sum() > 0)
            stimulus_counts[s_type] = stimulus_counts[s_type] + count
            stimulus_counts_rois[s_type] = stimulus_counts_rois[s_type] + count * rois_count

    return spontaneous_counts, stimulus_counts, stimulus_counts_rois


def get_rois_count(rois_per_sweep):
    roi_count_per_sweep = dict()
    roi_count_total = 0
    for sw in rois_per_sweep:
        rois = rois_per_sweep[sw].shape[0]
        roi_count_per_sweep[sw] = rois
        roi_count_total += rois

    return roi_count_per_sweep, roi_count_total


def responses_to_spontaneous_motor(scoring, score_th):
    results = dict()
    results['count'] = 0
    results['responsive'] = list()
    results['unresponsive'] = list()
    results['response_prob'] = list()
    results['scores'] = list()

    rois = scoring['roi'].unique()
    for r in rois:
        roi_data = scoring[scoring['roi'] == int(r)]
        roi_scores = roi_data[roi_data['reg'] == 'motor_spontaneous']['score']
        roi_mean_score = roi_scores.mean()
        results['scores'].append(float(roi_mean_score))
        if roi_mean_score >= score_th:
            results['count'] += 1
            results['responsive'].append(int(r))
        else:
            results['unresponsive'].append(int(r))

        roi_prob = (roi_scores >= score_th).sum() / len(roi_scores)
        results['response_prob'].append(float(roi_prob))

    return results


def main():
    # Definitions:
    # swim bout: One burst of VR activity
    # motor event: swim bouts that are closer than 10 seconds are combined to a motor event

    ca_labels, vr_events, sw_rois, linear_scoring_results = load_data()

    # Get ROIs names per sweep
    rois_per_sweep = get_rois_per_sweep(ca_labels)

    # Get number of spontaneous swims bouts per sweep
    spontaneous_swims_per_sweep, stimulus_swims_total, stimulus_swims_rois = get_count_spontaneous_swims(vr_events, rois_per_sweep)

    # Get ROIs count per sweep
    roi_count_per_sweep, roi_count_total = get_rois_count(rois_per_sweep)

    # SPONTANEOUS MOTOR EVENTS
    # Number of ROIs with sign. response to spontaneous motor events (mean score > th)
    # Response Prob. to spontaneous motor events
    # Mean Scores for each ROI

    sp_motor_results = responses_to_spontaneous_motor(linear_scoring_results, score_th=0.09)

    plt.plot(sp_motor_results['responsive'], np.zeros_like(sp_motor_results['responsive']), 'ro')
    plt.plot(sp_motor_results['unresponsive'], np.zeros_like(sp_motor_results['unresponsive']), 'bo')
    plt.show()

    print(sp_motor_results['responsive'])
    print('')
    print(sp_motor_results['unresponsive'])

    plt.hist(sp_motor_results['scores'], bins=50)
    plt.show()

    # rois = linear_scoring_results['roi'].unique()
    # k = 80
    #
    # idx = linear_scoring_results['roi'] == int(rois[k])
    # roi_data = linear_scoring_results[idx]
    # idx2 = roi_data['reg'] == 'motor_spontaneous'
    #
    # print('')
    # print(f'ROI: {rois[k]}')
    # print(roi_data[idx2]['score'])
    # print('')
    # print(roi_data[idx2]['score'].mean())

    # linear_scoring_results[linear_scoring_results['roi'] == 266]

    embed()
    exit()


if __name__ == "__main__":
    main()
