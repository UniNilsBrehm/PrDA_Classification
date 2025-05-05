#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from utils import load_hdf5_as_dict
from IPython import embed


def load_data():
    scoring_data = pd.read_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv')

    # Remove multiple entries: Motor Spontaneous and Moving Target 01 and 02
    # There are mean values for these already present
    idx = scoring_data['reg'].isin(['motor_spontaneous', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
    scoring_data_filtered = scoring_data[~idx].reset_index(drop=True)

    # Pivot data frame to get a long table format for the Scores
    values = 'score'
    df = scoring_data_filtered.pivot(index=['roi', 'sw'], columns='reg', values=values).reset_index()

    # Rename the columns to remove the _MEAN name
    df = df.rename(columns={
        'moving_target_MEAN': 'moving_target',
        'motor_spontaneous_MEAN': 'motor_spontaneous',
        'moving_target_motor_MEAN': 'moving_target_motor'
    })

    scores_org = df.drop(columns=['roi', 'sw']).reset_index(drop=True)
    scores_org.columns.name = None

    return scoring_data, scores_org


def load_vr_events():
    file_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/ventral_root/selected_sweeps_ventral_root_events.h5'
    vr_events = load_hdf5_as_dict(file_dir)
    return vr_events


def get_stimulus_types():
    stimulus_types = [
        # 'spontaneous',
        'moving_target_01',
        'moving_target_02',
        'grating_appears',
        'grating_disappears',
        'grating_0',
        'grating_180',
        'bright_loom',
        'dark_loom',
        'bright_flash',
        'dark_flash',
    ]
    return stimulus_types


def get_classes():
    # 12 classes
    logic_classes = [
        [1, 1, 1, 1],  # 1
        [0, 1, 0, 1],  # 2
        [0, 0, 0, 1],  # 3
        [1, 0, 0, 1],  # 4
        [1, 0, 1, 1],  # 5
        [1, 0, 1, 0],  # 6
        [1, 1, 1, 0],  # 7
        [1, 1, 0, 0],  # 8
        [0, 1, 0, 0],  # 9
        [1, 0, 0, 0],  # 10
        [1, 1, 0, 1],  # 11
        [0, 0, 0, 0]   # 12
    ]
    return logic_classes


def classifier(class_df, k):
    if k[1] == 1:
        idx = (class_df['sme'] == k[0]) & (class_df['clean_stimulus'] == k[1]) & (class_df['a_sme'] == k[2]) & (class_df['a_clean'] == k[3])
    else:
        # print('NO CLEAN STIMULUS')
        idx = (class_df['sme'] == k[0]) & (class_df['clean_stimulus'] == k[1]) & (class_df['a_sme'] == k[2]) & (class_df['a_s'] == k[3])

    class_members = class_df[idx]
    return class_members


def assign_classes(class_df):
    class_logics = get_classes()
    res = dict()
    rois = dict()
    class_data = dict()
    class_rois = list()
    k = 1
    t = 0
    for cl in class_logics:
        c = classifier(class_df, k=cl)
        class_data[k] = c
        res[k] = c.shape[0]
        rois[k] = c['roi'].reset_index(drop=True)
        class_rois.extend(c['roi'].to_list())
        print(f'Class {k}: {res[k]} cells')
        k += 1
        t = t + c.shape[0]

    all_rois = class_df['roi'].to_list()
    missing_rois = list(set(all_rois) - set(class_rois))
    print(f'Not classified ROIs: {missing_rois} cells')
    return class_data, res, rois


def create_classification_matrix(s, sw_vr_events, stimulus_types, roi, sw, score_th=0.1, min_clean_stimuli=2, sme_limit=3):
    motor_stimulus = dict()
    for s_type in stimulus_types:
        s_count = (sw_vr_events == s_type).sum()
        if s_count >= 1:
            motor_stimulus[s_type] = 1
        else:
            motor_stimulus[s_type] = 0

    motor_stimulus = pd.Series(motor_stimulus)

    # Get the corresponding scores
    mean_score = float(s[s['reg'].isin(motor_stimulus.index)]['score'].mean())
    max_score = float(s[s['reg'].isin(motor_stimulus.index)]['score'].max())

    # if mean_score >= score_th:
    if max_score >= score_th:

        a_s = 1
    else:
        a_s = 0

    # Clean Stimuli (without motor events)
    clean_stimulus = motor_stimulus[motor_stimulus == 0].index
    clean_stimulus_label = ', '.join(clean_stimulus.tolist())
    if len(clean_stimulus) >= min_clean_stimuli:  # There must be at least x clean stimuli
        clean_stimulus_status = 1
        # clean_score = float(s[s['reg'].isin(clean_stimulus)]['score'].mean())
        clean_score = float(s[s['reg'].isin(clean_stimulus)]['score'].max())
    else:
        clean_stimulus_status = 0
        clean_score = 0

    if clean_score >= score_th:
        a_clean = 1
    else:
        a_clean = 0

    # Dirty Stimuli (with motor events)
    dirty_stimulus = motor_stimulus[motor_stimulus == 1].index
    if len(dirty_stimulus) > 0:
        # dirty_stimulus_status = 1
        dirty_score = float(s[s['reg'].isin(dirty_stimulus)]['score'].mean())
    else:
        # dirty_stimulus_status = 0
        dirty_score = 0

    if dirty_score >= score_th:
        a_dirty = 1
    else:
        a_dirty = 0

    # spontaneous_motor_score = s[s['reg'] == 'motor_spontaneous_MEAN']['score'].item()
    sme_all_scores = s[s['reg'] == 'motor_spontaneous']['score']
    spontaneous_motor_score = s[s['reg'] == 'motor_spontaneous']['score'].max()
    # Check for enough spontaneous motor events

    sme_count = s[s['reg'] == 'motor_spontaneous'].shape[0]
    if sme_count >= sme_limit:
        sme_class = 1
        if spontaneous_motor_score >= score_th:
            a_sme = 1
            sme_prob = (sme_all_scores > score_th).sum() / sme_all_scores.shape[0]
        else:
            a_sme = 0
            sme_prob = 0
    else:
        sme_class = 0
        a_sme = 0
        sme_prob = 0

    row = [
        int(roi), sw, int(s['x_position'].unique()[0]), int(s['y_position'].unique()[0]),
        int(s['z_position'].unique()[0]),
        sme_class, clean_stimulus_status, a_sme, a_s, a_clean, a_dirty,
        mean_score, clean_score, dirty_score, spontaneous_motor_score, clean_stimulus_label, sme_prob
    ]
    row.extend(1 - motor_stimulus.values)

    return row


def main():
    scores_raw, scores = load_data()
    vr_events = load_vr_events()
    stimulus_types = get_stimulus_types()

    sme_limit = 3
    score_th = 0.1
    min_clean_stimuli = 2
    classification_matrix = list()
    roi_names = scores_raw['roi'].unique()

    for roi in roi_names:
        s = scores_raw[scores_raw['roi'] == roi]
        sw = s['sw'].unique()[0]
        sw_vr_events = vr_events[sw]['stimulus']

        row = create_classification_matrix(
            s=s, sw_vr_events=sw_vr_events, roi=roi, sw=sw, stimulus_types=stimulus_types,
            score_th=score_th, min_clean_stimuli=min_clean_stimuli, sme_limit=sme_limit
        )
        classification_matrix.append(row)

    labels = [
        'roi', 'sw', 'x_pos', 'y_pos', 'z_pos', 'sme', 'clean_stimulus', 'a_sme', 'a_s', 'a_clean', 'a_dirty',
        'score_mean', 'score_clean', 'score_dirty', 'score_sme', 'clean_s_types', 'sme_prob'
    ]
    labels.extend(stimulus_types)

    class_df = pd.DataFrame(classification_matrix, columns=labels)

    # Assign Classes
    class_data, counts, rois = assign_classes(class_df)

    member_counts = pd.DataFrame(list(counts.items()), columns=['Class', 'Count'])

    # Get number of sign. dirty scores
    a = class_df[class_df['roi'].isin(rois[11])]['score_dirty']
    idx = a >= score_th
    print(f'{idx.sum()} / {idx.shape[0]} have sign. dirty score')

    # Pie Plot
    keep_keys = [1, 2, 7]
    # Separate the kept and the rest
    df_keep = member_counts[member_counts['Class'].isin(keep_keys)]
    rest_sum = member_counts[~member_counts['Class'].isin(keep_keys)]['Count'].sum()

    # Create a new DataFrame for plotting
    df_plot = df_keep.copy()
    df_plot.loc[len(df_plot)] = ['rest', rest_sum]  # add 'rest' row

    # Custom function to show both percentage and count
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'

        return my_autopct

    fig, axs = plt.subplots()
    axs.pie(df_plot['Count'], labels=df_plot['Class'], startangle=90, autopct=make_autopct(df_plot['Count']),)
    plt.show()

    embed()
    exit()
    # Look at it
    k = [1, 1, 1, 1]
    idx = (class_df['sme'] == k[0]) & (class_df['clean_stimulus'] == k[1]) & (class_df['a_sme'] == k[2]) & (class_df['a_clean'] == k[3])
    idx = (class_df['clean_stimulus'] == 1)

    idx = (class_df['sme'] == 1)
    with_sme = class_df[idx]

    plt.hist(with_sme['sme_prob'], bins=10)
    plt.show()

    idx = (class_df['a_sme'] == 1)
    sign_motor = class_df[idx]
    print(f'sign. Motor: {sign_motor.shape[0]} from {with_sme.shape[0]} cells with spont. motor events (>={sme_limit})')


    idx = with_sme['sme_prob'] >= 0.5
    a = with_sme[idx]

    # labels_drop = [
    #     'roi', 'sw', 'x_pos', 'y_pos', 'z_pos', 'sme', 'clean_stimulus', 'a_sme', 'a_s', 'a_clean', 'a_dirty',
    #     'score_mean', 'score_clean', 'score_dirty', 'score_sme', 'clean_s_types'
    # ]
    #
    # stimulus_matrix = class_df.drop(columns=labels_drop)
    # plt.pcolormesh(stimulus_matrix.to_numpy())
    # # Set the x-ticks at the center of each cell
    # plt.xticks(
    #     ticks=np.arange(stimulus_matrix.shape[1]) + 0.5,  # center of each cell
    #     labels=stimulus_matrix.columns,
    #     rotation=0  # optional: rotate labels for readability
    # )
    # plt.show()


if __name__ == "__main__":
    main()
