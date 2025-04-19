#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
import numpy as np
import pandas as pd
from utils import load_hdf5_as_dict, calcium_impulse_response, get_rois_per_sweep, create_regressors_from_binary, \
    get_rois_of_one_sweep, save_dat_frame_as_hdf5
from config import Config
from IPython import embed
import statsmodels.api as sm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'linear_scoring_analysis.py'

Score the response of ROIs to different stimuli and motor events using a linear regression model.
Score = R² * slope


Result: 
    - /data/linear_scoring_results.csv'
...

Requirements:


Nils Brehm  -  2025
"""


def get_spontaneous_events(vr_events, rois_per_sweep):
    sp_events = dict()
    number_sp_events = pd.DataFrame()
    rois_with_sp = 0
    for sw in vr_events:
        idx = vr_events[sw]['stimulus'] == 0
        sp_events[sw] = vr_events[sw][idx]
        event_count = sp_events[sw].shape[0]
        if event_count > 0:
            rois_with_sp = rois_with_sp + rois_per_sweep[sw].shape[0]
        number_sp_events[sw] = [event_count]

        # print(f'{sw}: {sp_events[sw].shape[0]} spont. swims')
    return sp_events, number_sp_events, rois_with_sp


def find_best_lag_cross_correlation(data_trace, design_matrix, lag_start, lag_end, plot=False):
    corr = np.correlate(data_trace, design_matrix.iloc[:, 1], mode='full')
    # Number of samples in the original signals
    n = len(data_trace)  # Assumes both signals are the same length

    # Create an array of lag values. Lags range from -(n-1) to (n-1)
    corr_lags = np.arange(-n + 1, n)

    # Find the index of the maximum correlation value
    max_index = np.argmax(corr)

    corr_best_lag = corr_lags[max_index]

    if -lag_start < corr_best_lag <= lag_end:
        result = corr_best_lag
    else:
        result = 0

    if plot:
        fig, axs = plt.subplots()
        axs.plot(corr_lags, corr, marker='o')
        axs.set_xlabel('Lag')
        axs.set_ylabel('Correlation')
        axs.axvline(x=corr_best_lag, color='r', linestyle='--', label=f'Best lag = {corr_best_lag}')
        axs.legend()
        plt.show()

    return result, corr_best_lag


def merge_events_closing(trace, max_gap):
    from scipy.ndimage import binary_closing

    trace = np.array(trace)
    # Create a structuring element that covers gaps of size max_gap
    structure = np.ones(max_gap * 2 + 1)

    # binary_closing will fill gaps that are smaller than the structuring element
    closed = binary_closing(trace, structure=structure)
    return closed.astype(int)


def get_event_onset_offset(data):
    """
    Detects the onset and offset indices of events in a merged binary trace.

    Parameters:
        data (array-like): Binary trace (numpy array or list of 0's and 1's).

    Returns:
        onsets (numpy.ndarray): Indices where events begin.
        offsets (numpy.ndarray): Indices where events end.
    """
    data = np.asarray(data)
    # Compute the difference between consecutive elements
    d = np.diff(data.astype(int))

    # Onset: Transition from 0 to 1 (difference equals +1)
    onsets = np.where(d == 1)[0] + 1  # +1 to account for the shift due to diff

    # Offset: Transition from 1 to 0 (difference equals -1)
    offsets = np.where(d == -1)[0]

    # If the signal starts with a 1, then index 0 is an onset.
    if data[0] == 1:
        onsets = np.insert(onsets, 0, 0)

    # If the signal ends with a 1, then the last index is an offset.
    if data[-1] == 1:
        offsets = np.append(offsets, len(data) - 1)

    return onsets, offsets


def linear_regression_model(data_trace, design_matrix, lag, debugging=False):
    # Shift the regressors by the current lag.
    # Note: The shift() function shifts the index by the given number of periods.
    x_shifted = design_matrix.shift(lag)

    # Combine y and shifted X into a single DataFrame and drop rows with NaNs.
    df = pd.concat([data_trace, x_shifted], axis=1).dropna()
    # df = pd.concat([data_trace, x_shifted], axis=1).fillna(0)

    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1:]

    # Compute LM-OLS Model
    model = sm.OLS(y_clean, x_clean).fit()
    r2 = model.rsquared
    cf = model.params.iloc[1]

    score = cf * r2
    # if score < 0:
    #     score = 0
    # if r2 <= 0:
    #     score = 0
    # if cf < 0:
    #     score = 0

    scoring_results = {
        'r2': r2,
        'cf': float(cf),
        'score': float(score)
    }
    if debugging:
        return scoring_results, x_clean, y_clean
    else:
        return scoring_results


def linear_scoring(ca_data, motor_events, visual_events, ca_sampling_rate, stimulus_onsets, stimulus_offsets, settings):
    # Generate Calcium Impulse Response Function
    _, cif = calcium_impulse_response(tau_rise=settings['tau_rise'], tau_decay=settings['tau_decay'], amplitude=1.0,
                                      sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)

    # Dummy code for strange cases
    dummy_code = -1

    # Collect Motor Events
    motor_spontaneous = motor_events['spontaneous']

    # Merge Motor Events that are too close
    max_gap_secs = settings['max_gap_secs']
    motor_spontaneous_events = merge_events_closing(motor_spontaneous, max_gap=int(max_gap_secs * ca_sampling_rate))

    # Find onset and offsets of motor events
    motor_spontaneous_idx = get_event_onset_offset(motor_spontaneous_events)

    # Generate calcium regressors
    motor_spontaneous_regressor = create_regressors_from_binary(motor_spontaneous_events, cif, delta=True, norm=True)
    visual_regressors = pd.DataFrame()
    motor_stimulus_regressors = pd.DataFrame()
    for k in visual_events:
        visual_regressors[k] = create_regressors_from_binary(visual_events[k], cif, delta=True, norm=True)
        motor_merged = merge_events_closing(motor_events[k], max_gap=int(max_gap_secs * ca_sampling_rate))
        motor_stimulus_regressors[k] = create_regressors_from_binary(motor_merged, cif, delta=True, norm=True)

    # time_before = settings['time_before']
    # time_after = settings['time_after']
    # time_before_samples = int(time_before * ca_sampling_rate)
    # time_after_samples = int(time_after * ca_sampling_rate)
    # lag_start_sec = settings['lag_start_sec']
    # lag_end_sec = settings['lag_end_sec']

    sweep_scoring_result = list()

    # Loop over each ROI (column in df)
    for roi in ca_data.columns:
        # roi = '31'
        scores_list = list()
        r2_list = list()
        cf_list = list()
        reg_list = list()
        lags_list = list()

        y_trace = ca_data[roi]  # Get calcium trace
        # Linear Scoring for Visual Stimulus regressors
        for s_k, e_k in zip(stimulus_onsets, stimulus_offsets):
            # s_k and e_k are both the stimulus name (key)
            # s_k = 'dark_flash'
            # e_k = 'dark_flash'
            # ==========================================================================================================
            # VISUAL REGRESSORS
            time_before = settings['time_window'][s_k][0]
            time_after = settings['time_window'][s_k][1]
            lag_start_sec = settings['lag_window'][s_k][0]
            lag_end_sec = settings['lag_window'][s_k][1]

            # Create window for cutting out
            s_idx = int((stimulus_onsets[s_k].item() - time_before) * ca_sampling_rate)
            e_idx = int((stimulus_offsets[e_k].item() + time_after) * ca_sampling_rate)

            # Cut out stimulus window
            y_segment = y_trace[s_idx:e_idx].reset_index(drop=True)
            reg = visual_regressors[s_k][s_idx:e_idx].reset_index(drop=True)

            # Create design matrix and add Intercept
            design_matrix = sm.add_constant(reg)

            # Linear Regression with Time Alignment
            best_lag, cbl = find_best_lag_cross_correlation(
                y_segment, design_matrix, int(lag_start_sec*ca_sampling_rate), int(lag_end_sec*ca_sampling_rate)
            )
            lm_result = linear_regression_model(y_segment, design_matrix, best_lag)

            best_lag_secs = best_lag / ca_sampling_rate
            scores_list.append(lm_result['score'])
            r2_list.append(lm_result['r2'])
            cf_list.append(lm_result['cf'])
            reg_list.append(s_k)
            lags_list.append(best_lag_secs)

            # ==========================================================================================================
            # MOTOR EVENTS
            # Look for Motor Events during Stimulus Window
            motor_reg = motor_stimulus_regressors[s_k][s_idx:e_idx].reset_index(drop=True)

            # Create design matrix and add Intercept
            motor_design_matrix = sm.add_constant(motor_reg)

            # Linear Regression with Time Alignment
            motor_best_lag, cbl = find_best_lag_cross_correlation(
                y_segment, motor_design_matrix,
                int(lag_start_sec*ca_sampling_rate), int(lag_end_sec*ca_sampling_rate),
                plot=False
            )
            motor_lm_result = linear_regression_model(y_segment, motor_design_matrix, motor_best_lag)

            motor_best_lag_secs = best_lag / ca_sampling_rate

            scores_list.append(motor_lm_result['score'])
            r2_list.append(motor_lm_result['r2'])
            cf_list.append(motor_lm_result['cf'])
            reg_list.append(f'{s_k}_motor')
            lags_list.append(motor_best_lag_secs)

        # ==============================================================================================================
        # Motor Spontaneous Events
        if motor_spontaneous_idx[0].shape[0] > 0:
            for s_idx, e_idx in zip(motor_spontaneous_idx[0], motor_spontaneous_idx[1]):
                # Cut out stimulus window
                time_before = settings['time_window']['motor_spontaneous'][0]
                time_after = settings['time_window']['motor_spontaneous'][1]
                lag_start_sec = settings['lag_window']['motor_spontaneous'][0]
                lag_end_sec = settings['lag_window']['motor_spontaneous'][1]

                start_idx = s_idx - int(time_before * ca_sampling_rate)
                end_idx = e_idx + int(time_after * ca_sampling_rate)

                y_segment = y_trace[start_idx:end_idx].reset_index(drop=True)
                reg = pd.DataFrame(motor_spontaneous_regressor[start_idx:end_idx])

                # Create design matrix and add Intercept
                design_matrix = sm.add_constant(reg)

                # Linear Regression with Time Alignment
                best_lag, cbl = find_best_lag_cross_correlation(
                    y_segment, design_matrix,
                    int(lag_start_sec * ca_sampling_rate), int(lag_end_sec * ca_sampling_rate), plot=False
                )
                lm_result = linear_regression_model(y_segment, design_matrix, best_lag)

                best_lag_secs = best_lag / ca_sampling_rate
                scores_list.append(lm_result['score'])
                r2_list.append(lm_result['r2'])
                cf_list.append(lm_result['cf'])
                reg_list.append('motor_spontaneous')
                lags_list.append(best_lag_secs)

        metrics = ['score', 'r2', 'coeff', 'lag_sec']
        scoring = pd.DataFrame([scores_list, r2_list, cf_list, lags_list]).transpose()
        scoring.columns = metrics
        scoring['reg'] = reg_list

        # Moving Targets are presented twice. So take the mean
        idx_1 = scoring['reg'] == 'moving_target_01'
        idx_2 = scoring['reg'] == 'moving_target_02'
        mt_means = pd.DataFrame([scoring[idx_1+idx_2][metrics].mean().to_numpy()], columns=metrics)
        mt_means['reg'] = 'moving_target_MEAN'
        scoring = pd.concat([scoring, mt_means])

        # Same for the Motor Regressor
        idx_1 = scoring['reg'] == 'moving_target_01_motor'
        idx_2 = scoring['reg'] == 'moving_target_02_motor'
        motor_mt_means = pd.DataFrame([scoring[idx_1+idx_2][metrics].mean().to_numpy()], columns=metrics)
        motor_mt_means['reg'] = 'moving_target_motor_MEAN'
        scoring = pd.concat([scoring, motor_mt_means])

        # Motor Spontaneous Mean
        idx_sp = scoring['reg'] == 'motor_spontaneous'
        if motor_spontaneous_idx[0].shape[0] > 0:
            motor_sp_means = pd.DataFrame([scoring[idx_sp][metrics].mean().to_numpy()], columns=metrics)
        else:
            motor_sp_means = pd.DataFrame([np.zeros(len(metrics)) + dummy_code], columns=metrics)

        motor_sp_means['reg'] = 'motor_spontaneous_MEAN'
        scoring = pd.concat([scoring, motor_sp_means])

        # Add ROI name
        scoring['roi'] = roi
        scoring = scoring.reset_index(drop=True)
        sweep_scoring_result.append(scoring)
    results_df = pd.concat(sweep_scoring_result)
    return results_df


def compute_linear_scoring(settings):
    from time import perf_counter
    t0 = perf_counter()
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    # ca_df_f = pd.read_csv(Config.ca_df_f_no_motor_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    ca_labels = pd.read_csv(Config.ca_labels_file)
    # vr_events = load_hdf5_as_dict(Config.vr_events_file)
    rois_per_sweep = get_rois_per_sweep(ca_labels)
    # sp_events, number_sp_events, rois_with_sp = get_spontaneous_events(vr_events, rois_per_sweep)
    # rois_without_sp = ca_labels.shape[1] - rois_with_sp
    # stimulus_traces = pd.read_csv(Config.stimulus_traces_file, index_col=0)

    vr_binaries = load_hdf5_as_dict(Config.vr_all_binaries_file)

    for k in rois_per_sweep:
        print(k)

    all_results = list()
    for sw in rois_per_sweep:
        # sw = 'sw_02'
        # Get all the ROIs of this sweep
        sw_rois = get_rois_of_one_sweep(sw, rois_per_sweep)

        # Get Binaries for all Stimuli
        sw_stimulus_binaries = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_binaries.csv', index_col=0)
        sw_stimulus_onsets = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_onsets.csv', index_col=0)
        sw_stimulus_offsets = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_offsets.csv', index_col=0)

        # Get Spontaneous and Stimulus Motor Events for this sweep
        motor_events = vr_binaries[sw]

        # Get all ca data traces from all ROIs
        sw_ca_traces = ca_df_f[sw_rois]

        print(f'Sweep: {sw}')
        scores = linear_scoring(
            sw_ca_traces, motor_events, sw_stimulus_binaries, ca_sampling_rate,
            sw_stimulus_onsets, sw_stimulus_offsets, settings
        )

        # Add sweep name
        scores['sw'] = sw

        # Add anatomical coordinates
        anatomical_positions_x = list()
        anatomical_positions_y = list()
        z_planes = list()
        for roi in scores['roi'].unique():
            n = scores[scores['roi'] == roi].shape[0]
            pos = ca_labels[roi].iloc[3]
            x_coordinate, y_coordinate = map(int, pos.split('-'))

            anatomical_positions_x.extend([x_coordinate] * n)
            anatomical_positions_y.extend([y_coordinate] * n)

            z_val = int(ca_labels[roi].iloc[4])
            z_planes.extend([z_val] * n)

        scores['x_position'] = anatomical_positions_x
        scores['y_position'] = anatomical_positions_y
        scores['z_position'] = z_planes
        scores = scores.reset_index(drop=True)
        # print(f'This took: {(perf_counter() - t1) / 60:.3f} minutes.')

        # Put everything together
        all_results.append(scores)

    results_df = pd.concat(all_results).reset_index(drop=True)
    # results_df.to_csv(Config.linear_scoring_file, index=False)
    results_df.to_csv(f'{Config.BASE_DIR}/data/linear_scoring_results.csv', index=False)

    print('==== FINISHED LINEAR SCORING ====')
    print(f'This took: {(perf_counter() - t0) / 60:.3f} minutes.')


def permutation_null_distribution2(settings):
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_df_f = ca_df_f.to_numpy()
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    # linear_scoring_file = f'{Config.BASE_DIR}/data/linear_scoring_results.csv'
    # linear_scoring_results = pd.read_csv(linear_scoring_file)
    data_size = ca_df_f.shape[0]
    # ca_time_axis = np.linspace(0, ca_df_f.shape[0]/ca_sampling_rate, ca_df_f.shape[0])
    # Generate Calcium Impulse Response Function
    _, cif = calcium_impulse_response(tau_rise=settings['tau_rise'], tau_decay=settings['tau_decay'], amplitude=1.0,
                                      sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)

    # Run n permutations
    null_dist = list()
    for k in range(settings['n_permutations']):
        # Loop over all ROIs
        permutation_round = list()
        for ca_trace in ca_df_f.T:
            # Get a random point in time
            time_point = np.random.randint(20, data_size-20)
            time_before = settings['time_before']
            time_after = settings['time_after']
            time_before_samples = int(time_before * ca_sampling_rate)
            time_after_samples = int(time_after * ca_sampling_rate)

            # Create Regressor
            random_binary = np.zeros(data_size)
            random_binary[time_point] = 1
            reg_trace = create_regressors_from_binary(random_binary, cif, delta=True, norm=True)

            # Cut out window around ca trace and regressor
            start_idx = time_point - time_before_samples
            end_idx = time_point + time_after_samples
            y_segment = ca_trace[start_idx:end_idx]
            reg = reg_trace[start_idx:end_idx]

            # Create design matrix and add Intercept
            design_matrix = sm.add_constant(reg)

            # Linear Regression
            # Compute LM-OLS Model
            model = sm.OLS(y_segment, design_matrix).fit()
            r2 = model.rsquared
            cf = model.params[1]
            score = r2 * cf
            permutation_round.append(float(score))

            # if score > 1.0:
            #     plt.plot(y_segment, 'k')
            #     plt.plot(reg, 'r')
            #     plt.show()
            #     embed()
            #     exit()

        null_dist.extend(permutation_round)
        print(f'{k+1} / {settings["n_permutations"]}', end='\r')


def single_permutation(ca_df_f, cif, ca_sampling_rate, settings, data_size):
    permutation_round = []
    for ca_trace in ca_df_f.T:
        time_point = np.random.randint(20, data_size - 20)
        time_before_samples = int(settings['time_before'] * ca_sampling_rate)
        time_after_samples = int(settings['time_after'] * ca_sampling_rate)

        random_binary = np.zeros(data_size)
        random_binary[time_point] = 1
        reg_trace = create_regressors_from_binary(random_binary, cif, delta=True, norm=True)

        start_idx = time_point - time_before_samples
        end_idx = time_point + time_after_samples
        y_segment = ca_trace[start_idx:end_idx]
        reg = reg_trace[start_idx:end_idx]

        design_matrix = sm.add_constant(reg)
        model = sm.OLS(y_segment, design_matrix).fit()
        r2 = model.rsquared
        cf = model.params[1]
        score = r2 * cf
        permutation_round.append(float(score))

    return permutation_round


def permutation_null_distribution(settings):
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_df_f = ca_df_f.to_numpy()
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    data_size = ca_df_f.shape[0]

    _, cif = calcium_impulse_response(tau_rise=settings['tau_rise'], tau_decay=settings['tau_decay'],
                                      amplitude=1.0, sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)

    # Parallel execution over permutations
    results = Parallel(n_jobs=-1)(  # Use all cores
        delayed(single_permutation)(ca_df_f, cif, ca_sampling_rate, settings, data_size)
        for _ in tqdm(range(settings['n_permutations']))
    )

    # Flatten the list of lists
    null_dist = [score for sublist in results for score in sublist]
    return null_dist


def main():
    settings = {
        'max_gap_secs': 10,  # Merge Spontaneous Motor Events that are too close (seconds)
        # 'time_before': 5,  # Cutout: Time before event starts (seconds)
        # 'time_after': 20,  # Cutout: Time after event ends (seconds)
        # 'lag_start_sec': 5,  # Time shift for cross correlation to the left (seconds)
        # 'lag_end_sec': 5,  # Time shift for cross correlation to the right (seconds)

        'tau_rise': 3,  # Rise Time Constant for Calcium Impulse Response Function (seconds)
        'tau_decay': 6,  # Decay Time Constant for Calcium Impulse Response Function (seconds)

        'lag_window': {  # Time shift for cross correlation to the left / right (seconds)
            'moving_target_01': [2, 10],
            'moving_target_02': [2, 10],
            'grating_appears': [2, 5],
            'grating_disappears': [2, 5],
            'grating_0': [2, 10],
            'grating_180': [2, 10],
            'bright_loom': [2, 5],
            'dark_loom': [2, 5],
            'bright_flash': [2, 5],
            'dark_flash': [2, 5],
            'motor_spontaneous': [5, 5]
        },

        'time_window': {  # Cutout: Time before and after stimulus period for linear regression model
            'moving_target_01': [2, 5],  # window: -2 to 11 s (stim duration 6 s)
            'moving_target_02': [2, 5],
            'grating_appears': [2, 10],  # window: -2 to 11 s (stim duration 1 s)
            'grating_disappears': [2, 10],
            'grating_0': [2, 2],  # window: -2 to 12 s (stim duration 10 s)
            'grating_180': [2, 2],
            'bright_loom': [2, 10],  # window: -2 to 12 s (stim duration 2 s)
            'dark_loom': [2, 10],
            'bright_flash': [2, 10],  # window: -2 to 11 s (stim duration 1 s)
            'dark_flash': [2, 10],
            'motor_spontaneous': [2, 15]  # window: -2 to 15 s around swim onset
        }
    }
    compute_linear_scoring(settings)
    exit()

    # Null Distribution (Permutation/Bootstrapping Test)
    permutation_settings = {
        'n_permutations': 10000,
        'time_before': 2,  # Cutout: Time before event starts (seconds)
        'time_after': 15,  # Cutout: Time after event ends (seconds)

        'tau_rise': 3,  # Rise Time Constant for Calcium Impulse Response Function (seconds)
        'tau_decay': 6  # Decay Time Constant for Calcium Impulse Response Function (seconds)
    }
    # permutation_null_distribution2(permutation_settings)
    null_dist = permutation_null_distribution(permutation_settings)
    save_dat_frame_as_hdf5(f'{Config.BASE_DIR}/data/scores_null_distribution_10k_permutations.h5', pd.DataFrame(null_dist))
    print('==== FINISHED PERMUTATION NULL DISTRIBUTION ====')


if __name__ == "__main__":
    main()


# BACKUPS
# def subtract_spontaneous_baseline(df, motor_events, visual_events, ca_sampling_rate, sigma, width):
#     """
#     Subtracts the baseline spontaneous activity from calcium traces.
#
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#
#     Returns:
#     pd.DataFrame: Baseline-corrected calcium traces.
#     """
#     # Define spontaneous periods (where both motor and visual are zero)
#     # spontaneous_mask = (motor_events == 0) & (visual_events == 0)
#     combined_mask = motor_events + visual_events
#     combined_mask[combined_mask > 1] = 1
#
#     spontaneous_mask = broaden_binary_trace(combined_mask, width=width)
#     spontaneous_mask_smooth = binary_to_gaussian(spontaneous_mask, ca_sampling_rate, sigma=sigma)
#     spontaneous_mask_smooth = spontaneous_mask_smooth / np.max(spontaneous_mask_smooth)
#     # Compute baseline for each ROI
#     # baseline = df[spontaneous_mask].mean(axis=0)
#
#     # Subtract baseline from all time points
#     # df_corrected = df - baseline
#     df_corrected = df.mul(spontaneous_mask_smooth, axis=0)
#     return df_corrected
#
#
# def glm_variance_partitioning(df, motor_events, visual_events, ca_sampling_rate, num_permutations=100):
#     """
#     Uses a Generalized Linear Model (GLM) to partition variance between motor and visual events.
#
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#     ca_sampling_rate (float): Calcium imaging sampling rate.
#     num_permutations (int): Number of permutations for statistical significance testing.
#
#     Returns:
#     pd.DataFrame: Contains full model pseudo R², unique and shared variance, and p-values for each ROI.
#     """
#     # Generate Calcium Impulse Response Function
#     _, cif = calcium_impulse_response(tau_rise=1, tau_decay=3, amplitude=1.0, sampling_rate=ca_sampling_rate, threshold=1e-5, norm=False)
#
#     # Generate calcium regressors
#     motor_regressor = create_regressors_from_binary(motor_events, cif, delta=False, norm=False)
#     visual_regressor = create_regressors_from_binary(visual_events, cif, delta=False, norm=False)
#
#     # Subtract Baseline from Ca Traces
#     ca_data = subtract_spontaneous_baseline(df, motor_events, visual_events, ca_sampling_rate, sigma=10, width=5)
#
#     results = []
#
#     # Loop over each ROI (column in df)
#     for roi in ca_data.columns:
#         y = ca_data[roi].values  # Get calcium trace
#         y = np.clip(y, 1e-5, None)  # Ensure positive values for Poisson/Gamma
#
#         # Create Design Matrix: Motor + Visual with Intercept
#         X_full = np.column_stack((motor_regressor, visual_regressor))
#         X_full = sm.add_constant(X_full)  # Adds intercept term
#
#         # Null Model (Intercept Only)
#         X_null = np.ones((len(y), 1))
#         null_model = sm.GLM(y, X_null, family=sm.families.Poisson()).fit()
#         logL_null = null_model.llf  # Log-likelihood of null model
#
#         # Full Model (Motor + Visual)
#         full_model = sm.GLM(y, X_full, family=sm.families.Poisson()).fit()
#         logL_full = full_model.llf  # Log-likelihood of full model
#         r2_full = 1 - (logL_full / logL_null)  # McFadden's Pseudo R²
#
#         # Compute Partial R² for Motor (Unique Contribution)
#         X_visual_only = sm.add_constant(visual_regressor)
#         model_visual_only = sm.GLM(y, X_visual_only, family=sm.families.Poisson()).fit()
#         r2_motor_unique = max(r2_full - (1 - (model_visual_only.llf / logL_null)), 0)
#
#         # Compute Partial R² for Visual (Unique Contribution)
#         X_motor_only = sm.add_constant(motor_regressor)
#         model_motor_only = sm.GLM(y, X_motor_only, family=sm.families.Poisson()).fit()
#         r2_visual_unique = max(r2_full - (1 - (model_motor_only.llf / logL_null)), 0)
#
#         # Compute Shared Contribution
#         r2_shared = max(r2_full - r2_motor_unique - r2_visual_unique, 0)
#
#         # Compute Unexplained Variance
#         r2_unexplained = max(1 - r2_full, 0)
#
#         # Permutation testing
#         shuffled_scores = []
#         for _ in range(num_permutations):
#             shuffled_X = shuffle(X_full, random_state=None)
#             shuffled_model = sm.GLM(y, shuffled_X, family=sm.families.Poisson()).fit()
#             shuffled_logL_full = shuffled_model.llf
#             shuffled_r2 = 1 - (shuffled_logL_full / logL_null)
#             shuffled_scores.append(shuffled_r2)
#
#         # Compute p-value
#         p_value = np.sum(np.array(shuffled_scores) >= r2_full) / num_permutations
#
#         # Store results
#         results.append({
#             "ROI": roi,
#             "Pseudo R² Full Model": r2_full,
#             "Pseudo R² Motor Unique": r2_motor_unique,
#             "Pseudo R² Visual Unique": r2_visual_unique,
#             "Pseudo R² Shared": r2_shared,
#             "Pseudo R² Unexplained": r2_unexplained,
#             "P-Value": p_value
#         })
#         return results
#
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#
#     return results_df


# def compute_spontaneous_mask(motor_events, visual_events, width):
#     motor = broaden_binary_trace(motor_events, width=width)
#     visual = broaden_binary_trace(visual_events, width=width)
#
#     spontaneous_mask = (motor == 0) & (visual == 0)
#
#     return spontaneous_mask


# def get_windowed_activity(df, event_binary, ca_sampling_rate, width_left, width_right):
#     """
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#
#     Returns:
#     pd.DataFrame: Baseline-corrected calcium traces.
#     """
#     # Define spontaneous periods (where both motor and visual are zero)
#
#     # Extend mask
#     event_mask = broaden_binary_trace_asymmetrical(
#         event_binary, left_width=int(ca_sampling_rate * width_left), right_width=int(ca_sampling_rate * width_right))
#
#     # Multiply Mask with data traces to set them to zero where mask is also zero
#     df_corrected = df.mul(event_mask, axis=0)
#
#     return df_corrected
#
#
# def cutout_windowed_activity(df, event_binary, event_regressor, ca_sampling_rate, width_left, width_right, stitch=False):
#     """
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     event_binary (np.array): Binary event trace.
#
#     Returns:
#     pd.DataFrame: stitched calcium traces.
#     """
#     # Define spontaneous periods (where both motor and visual are zero)
#
#     # Extend mask
#     event_mask = broaden_binary_trace_asymmetrical(
#         event_binary, left_width=int(ca_sampling_rate * width_left), right_width=int(ca_sampling_rate * width_right))
#     event_mask_bool = event_mask.astype('bool')
#     df_stitched = df.loc[event_mask_bool, :].reset_index(drop=True)
#     event_regressor_stitched = event_regressor[event_mask_bool]
#
#     # # Identify segments
#     # segments = []
#     # current_segment = []
#     #
#     # data = pd.concat([df, pd.DataFrame(event_binary, columns=['binary'])], axis=1)
#     #
#     # for i, row in data.iterrows():
#     #     if row['binary'] == 1:
#     #         current_segment.append(row)
#     #     else:
#     #         if current_segment:  # Save the previous segment if it's not empty
#     #             segments.append(pd.DataFrame(current_segment))
#     #             current_segment = []
#     #
#     # # Add last segment if it's not empty
#     # if current_segment:
#     #     segments.append(pd.DataFrame(current_segment))
#     #
#     # # Display each segment
#     # for idx, segment in enumerate(segments):
#     #     print(f"\nSegment {idx + 1}:\n", segment)
#
#     # roi = 2
#     # fig, axs = plt.subplots(2, 1)
#     # axs[0].plot(df_stitched.iloc[:, roi], 'k')
#     # axs[0].plot(event_regressor_stitched, 'r')
#     # axs[1].plot(df.iloc[:, roi], 'k')
#     # axs[1].plot(event_regressor, 'r')
#     # plt.show()
#
#     return df_stitched, event_regressor_stitched
#
#
# def get_spontaneous_activity2(df, motor_events, visual_events, ca_sampling_rate, sigma, width):
#     """
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#
#     Returns:
#     pd.DataFrame: Baseline-corrected calcium traces.
#     """
#     # Define spontaneous periods (where both motor and visual are zero)
#     combined_mask = motor_events + visual_events
#     combined_mask[combined_mask > 1] = 1
#
#     # Extend mask
#     spontaneous_mask = broaden_binary_trace(combined_mask, width=int(ca_sampling_rate * width))
#
#     if sigma > 0:
#         spontaneous_mask_smooth = binary_to_gaussian(spontaneous_mask, ca_sampling_rate, sigma=sigma)
#         spontaneous_mask_smooth = spontaneous_mask_smooth / np.max(spontaneous_mask_smooth)
#
#         # Invert mask
#         spontaneous_mask_final = 1 - spontaneous_mask_smooth
#     else:
#         spontaneous_mask_final = 1 - spontaneous_mask
#
#     # Subtract baseline from all time points
#     df_corrected = df.mul(spontaneous_mask_final, axis=0)
#
#     return df_corrected
#
#
# def get_spontaneous_activity(df, ca_sampling_rate):
#     # Low pass filter data to get estimate for spontaneous activity
#     from utils import filter_low_pass
#     estimate = filter_low_pass(df, cutoff=0.0025, fs=ca_sampling_rate, order=2)
#     estimate_df = pd.DataFrame(estimate, columns=df.columns)
#
#     # plt.plot(df.iloc[:, 0], 'k')
#     # plt.plot(estimate_df.iloc[:, 0], 'r')
#     # plt.show()
#
#     return estimate_df


# def multiple_regression_with_spontaneous(df, motor_events_spontaneous, motor_events_stimulus, visual_events, ca_sampling_rate, num_permutations=1000):
#     """
#     Performs multiple regression analysis to partition variance, adding a spontaneous activity regressor.
#
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#     ca_sampling_rate (float): Calcium imaging sampling rate.
#     num_permutations (int): Number of permutations for statistical significance testing.
#
#     Returns:
#     pd.DataFrame: Contains full model R², unique and shared variance, and p-values for each ROI.
#     """
#
#     # Generate Calcium Impulse Response Function
#     _, cif = calcium_impulse_response(tau_rise=2, tau_decay=7, amplitude=1.0, sampling_rate=ca_sampling_rate,
#                                       threshold=1e-5, norm=True)
#
#     motor_events_all = motor_events_stimulus + motor_events_spontaneous
#     motor_events_all[motor_events_all > 1] = 1
#
#     # Generate calcium regressors
#     motor_spontaneous_regressor = create_regressors_from_binary(motor_events_spontaneous, cif, delta=True, norm=False)
#     motor_stimulus_regressor = create_regressors_from_binary(motor_events_stimulus, cif, delta=True, norm=False)
#     visual_regressor = create_regressors_from_binary(visual_events, cif, delta=True, norm=False)
#
#     # ca_data = df / df.max(axis=0)
#     ca_data = df.copy()
#
#     # Cut out and stitch Data for Spontaneous-Motor Events
#     ca_data_motor_spontaneous_stitched, motor_spontaneous_regressor_stitched = cutout_windowed_activity(
#         ca_data, motor_events_spontaneous, motor_spontaneous_regressor, ca_sampling_rate, width_left=2, width_right=30
#     )
#
#     # Cut out and stitch Data for Stimulus-Motor Events
#     ca_data_motor_stimulus_stitched, motor_stimulus_regressor_stitched = cutout_windowed_activity(
#         ca_data, motor_events_stimulus, motor_stimulus_regressor, ca_sampling_rate, width_left=2, width_right=30
#     )
#
#     # Cut out and stitch Data for Visual Stimulus
#     ca_data_visual_stitched, visual_regressor_stitched = cutout_windowed_activity(
#         ca_data, visual_events, visual_regressor, ca_sampling_rate, width_left=2, width_right=30
#     )
#
#     ca_data_spontaneous_activity = get_spontaneous_activity(
#         ca_data, motor_events_all, visual_events, ca_sampling_rate, sigma=0, width=10
#     )
#
#     ca_data_spontaneous_motor = get_windowed_activity(
#         ca_data, motor_events_spontaneous, ca_sampling_rate, width_left=5, width_right=30
#     )
#
#     ca_data_stimulus_motor = get_windowed_activity(
#         ca_data, motor_events_stimulus, ca_sampling_rate, width_left=5, width_right=30
#     )
#
#     ca_data_visual = get_windowed_activity(
#         ca_data, visual_events, ca_sampling_rate, width_left=5, width_right=30
#     )
#
#     results = []
#     # Loop over each ROI (column in df)
#     for roi in ca_data.columns:
#         y = ca_data[roi].values  # Get calcium trace
#         spontaneous_component = ca_data_spontaneous_activity[roi].values
#
#         # Create Design Matrix: Motor + Visual + Spontaneous with Intercept
#         full_design_matrix = np.column_stack((motor_spontaneous_regressor, motor_stimulus_regressor, visual_regressor, spontaneous_component))
#         full_design_matrix = sm.add_constant(full_design_matrix)  # Adds intercept term
#
#         # Fit Multiple Regression Model
#         model = sm.OLS(y, full_design_matrix).fit()
#
#         # Compute Full Model R²
#         r2_full = model.rsquared
#
#         # Compute Semi-Partial R² for Motor Spontaneous
#         r2_no_spontaneous_motor = sm.OLS(y, sm.add_constant(
#             np.column_stack((motor_stimulus_regressor, visual_regressor, spontaneous_component)))).fit().rsquared
#         r2_motor_spontaneous_unique = max(r2_full - r2_no_spontaneous_motor, 0)
#
#         # Compute Semi-Partial R² for Motor Stimulus
#         r2_no_stimulus_motor = sm.OLS(y, sm.add_constant(
#             np.column_stack((motor_spontaneous_regressor, visual_regressor, spontaneous_component)))).fit().rsquared
#         r2_motor_stimulus_unique = max(r2_full - r2_no_stimulus_motor, 0)
#
#         # Compute Semi-Partial R² for Visual
#         r2_no_visual = sm.OLS(y, sm.add_constant(
#             np.column_stack((motor_spontaneous_regressor, motor_stimulus_regressor, spontaneous_component)))).fit().rsquared
#         r2_visual_unique = max(r2_full - r2_no_visual, 0)
#
#         # Compute Semi-Partial R² for Spontaneous
#         r2_no_spontaneous = sm.OLS(y, sm.add_constant(
#             np.column_stack((motor_spontaneous_regressor, motor_stimulus_regressor, visual_regressor)))).fit().rsquared
#         r2_spontaneous_unique = max(r2_full - r2_no_spontaneous, 0)
#
#         # Motor Shared
#         r2_no_motor = sm.OLS(y,
#                              sm.add_constant(np.column_stack((visual_regressor, spontaneous_component)))).fit().rsquared
#
#         # Total contribution from both motor regressors
#         r2_motor_group = max(r2_full - r2_no_motor, 0)
#
#         # Shared contribution of motor regressors (overlap, should be around zero)
#         r2_shared_motor = max(r2_motor_group - (r2_motor_spontaneous_unique + r2_motor_stimulus_unique), 0)
#
#         # Motor Stimulus and Visual Stimulus
#         # Combine regressors that are not motor_stimulus or visual.
#         reduced_design_motorstim_visual = np.column_stack((motor_spontaneous_regressor, spontaneous_component))
#         reduced_model_motorstim_visual = sm.OLS(y, sm.add_constant(reduced_design_motorstim_visual)).fit()
#         r2_no_motorstim_visual = reduced_model_motorstim_visual.rsquared
#
#         # Total contribution of motor_stimulus and visual (unique + shared)
#         r2_motorstim_visual_group = r2_full - r2_no_motorstim_visual
#
#         # Unique contribution for visual:
#         shared_motorstim_visual = max(r2_motorstim_visual_group - (r2_motor_stimulus_unique + r2_visual_unique), 0)
#
#         # Compute Shared Contribution
#         r2_shared = max(r2_full - r2_motor_spontaneous_unique - r2_motor_stimulus_unique - r2_visual_unique - r2_spontaneous_unique, 0)
#
#         # Compute Unexplained Variance
#         r2_unexplained = max(1 - r2_full, 0)
#
#         # Compute Scores
#         coeff_motor_spontaneous = model.params[1]
#         coeff_motor_stimulus = model.params[2]
#         coeff_visual = model.params[3]
#         coeff_spontaneous = model.params[4]
#
#         score_motor_spontaneous = coeff_motor_spontaneous * r2_motor_spontaneous_unique
#         score_motor_stimulus = coeff_motor_stimulus * r2_motor_stimulus_unique
#         score_visual = coeff_visual * r2_visual_unique
#         score_spontaneous = coeff_spontaneous * r2_spontaneous_unique
#
#         # Perform permutation test for significance
#         shuffled_scores = []
#         for _ in range(num_permutations):
#             shuffled_X = shuffle(full_design_matrix, random_state=None)
#             shuffled_model = sm.OLS(y, shuffled_X).fit()
#             shuffled_scores.append(shuffled_model.rsquared)
#
#         # Compute p-value
#         p_value = np.sum(np.array(shuffled_scores) >= r2_full) / num_permutations
#
#         # # For motor regressor (index 1)
#         # t_motor = model.tvalues[1]
#         # df_resid = model.df_resid
#         # partial_R2_motor_t = t_motor ** 2 / (t_motor ** 2 + df_resid)
#         # print("Partial R2 for motor regressor (using t-statistic):", partial_R2_motor_t)
#         #
#         # # For visual regressor (index 2)
#         # t_visual = model.tvalues[2]
#         # partial_R2_visual_t = t_visual ** 2 / (t_visual ** 2 + df_resid)
#         # print("Partial R2 for visual regressor (using t-statistic):", partial_R2_visual_t)
#         #
#         # # For spontaneous component (index 3)
#         # t_spont = model.tvalues[3]
#         # partial_R2_spont_t = t_spont ** 2 / (t_spont ** 2 + df_resid)
#         # print("Partial R2 for spontaneous component (using t-statistic):", partial_R2_spont_t)
#
#         y_stitched = ca_data_motor_spontaneous_stitched[roi].values
#         r2_stitched_motor_spontaneous = sm.OLS(y_stitched, motor_spontaneous_regressor_stitched).fit().rsquared
#
#         y_stitched = ca_data_motor_stimulus_stitched[roi].values
#         r2_stitched_motor_stimulus = sm.OLS(y_stitched, motor_stimulus_regressor_stitched).fit().rsquared
#
#         y_stitched = ca_data_visual_stitched[roi].values
#         r2_stitched_visual = sm.OLS(y_stitched, visual_regressor_stitched).fit().rsquared
#
#         y_cutout = ca_data_spontaneous_motor[roi].values  # Get calcium trace
#         r2_cutout_motor_spontaneous = sm.OLS(y_cutout, motor_spontaneous_regressor).fit().rsquared
#
#         y_cutout = ca_data_stimulus_motor[roi].values  # Get calcium trace
#         r2_cutout_motor_stimulus = sm.OLS(y_cutout, motor_stimulus_regressor).fit().rsquared
#
#         y_cutout = ca_data_visual[roi].values  # Get calcium trace
#         r2_cutout_visual = sm.OLS(y_cutout, visual_regressor).fit().rsquared
#
#         # SHARED CUTOUT MOTOR STIM AND VISUAL
#         y_cutout = ca_data_visual[roi].values  # Get calcium trace
#         design_matrix = sm.add_constant(np.column_stack((
#             motor_spontaneous_regressor, spontaneous_component, motor_stimulus_regressor,  visual_regressor
#         )))
#         model_full = sm.OLS(y_cutout, design_matrix).fit().rsquared
#
#         design_matrix = sm.add_constant(np.column_stack((motor_spontaneous_regressor, spontaneous_component)))
#         model_no = sm.OLS(y_cutout, design_matrix).fit().rsquared
#
#         design_matrix = sm.add_constant(np.column_stack((
#             motor_spontaneous_regressor, spontaneous_component, motor_stimulus_regressor
#         )))
#         model_no_vis = sm.OLS(y_cutout, design_matrix).fit().rsquared
#         vis_unique = model_full - model_no_vis
#
#         design_matrix = sm.add_constant(np.column_stack((
#             motor_spontaneous_regressor, spontaneous_component, visual_regressor
#         )))
#         model_no_motor_stim = sm.OLS(y_cutout, design_matrix).fit().rsquared
#         motor_stim_unique = model_full - model_no_motor_stim
#
#         total = model_full - model_no
#         shared = total - (vis_unique + motor_stim_unique)
#
#         # ===================
#         y_cutout = ca_data_visual[roi].values  # Get calcium trace
#         design_matrix = sm.add_constant(np.column_stack((visual_regressor, motor_stimulus_regressor)))
#         model_both = sm.OLS(y_cutout, design_matrix).fit().rsquared
#
#         model_vis = sm.OLS(y_cutout, visual_regressor).fit().rsquared
#         model_motor = sm.OLS(y_cutout, motor_stimulus_regressor).fit().rsquared
#
#         design_matrix = visual_regressor-motor_stimulus_regressor
#         design_matrix[design_matrix < 0] = 0
#         model_vis_minus_motor = sm.OLS(y_cutout, design_matrix).fit().rsquared
#
#         # Store results
#         results.append({
#             "ROI": roi,
#             "R² Full Model": float(r2_full),
#             "R² Motor Spontaneous Unique": float(r2_motor_spontaneous_unique),
#             "R² Motor Stimulus Unique": float(r2_motor_stimulus_unique),
#             "R² Motor Group": float(r2_motor_group),
#             "R² Shared Motor": float(r2_shared_motor),  # <-- new entry for shared motor variance
#             "R² Visual Unique": float(r2_visual_unique),
#             "R² Shared MotorStim+Visual": float(shared_motorstim_visual),
#             "R² Spontaneous Unique": float(r2_spontaneous_unique),
#             "R² Shared (All Regressors)": float(r2_shared),
#             "R² Unexplained": float(r2_unexplained),
#             # "SCORE Motor Spontaneous Unique": float(score_motor_spontaneous),
#             # "SCORE Motor Stimulus Unique": float(score_motor_stimulus),
#             # "SCORE Visual Unique": float(score_visual),
#             # "SCORE Spontaneous Unique": float(score_spontaneous),
#             "P-Value": float(p_value)
#         })
#
#         results1 = {
#             "R² Full Model": float(r2_full),
#             "R² Motor Spontaneous Unique": float(r2_motor_spontaneous_unique),
#             "R² Motor Stimulus Unique": float(r2_motor_stimulus_unique),
#             "R² Motor Group": float(r2_motor_group),
#             "R² Shared Motor": float(r2_shared_motor),  # <-- new entry for shared motor variance
#             "R² Visual Unique": float(r2_visual_unique),
#             "R² Shared MotorStim+Visual": float(shared_motorstim_visual),
#             "R² Spontaneous Unique": float(r2_spontaneous_unique),
#             "R² Shared (All Regressors)": float(r2_shared),
#             "R² Unexplained": float(r2_unexplained),
#             "P-Value": float(p_value)
#         }
#
#         results2 = {
#             "R² Cutout Motor Spontaneous": float(r2_cutout_motor_spontaneous),
#             "R² Cutout Motor Stimulus": float(r2_cutout_motor_stimulus),
#             "R² Cutout Visual": float(r2_cutout_visual),
#
#             "R² Stitched Motor Spontaneous": float(r2_stitched_motor_spontaneous),
#             "R² Stitched Motor Stimulus": float(r2_stitched_motor_stimulus),
#             "R² Stitched Visual": float(r2_stitched_visual),
#
#         }
#
#         # Detect Peaks in Ca
#         from scipy.signal import find_peaks
#         peaks, _ = find_peaks(y, height=np.mean(y) + 2 * np.std(y), distance=5)
#
#         print('')
#         print(f'ROI: {roi}')
#         for k in results1:
#             print(f'{k}. {results1[k]:.4f}')
#
#         print('================================')
#
#         for k in results2:
#             print(f'{k}. {results2[k]:.4f}')
#
#         print('')
#         print('')
#
#         plt.plot(y, 'k')
#         plt.plot(peaks, y[peaks], 'rx')
#         plt.plot(motor_spontaneous_regressor, 'r')
#         plt.plot(motor_stimulus_regressor, 'tab:orange')
#         plt.plot(visual_regressor, 'g')
#         plt.plot(spontaneous_component, 'gray', ls='--')
#         plt.show()
#
#         embed()
#         exit()
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#
#     return results_df



# def linear_regression_with_time_alignment(data_trace, design_matrix, lag_start, lag_end, center=False, tag='S'):
#     possible_lags = np.arange(-lag_start, lag_end+1, 1)
#     # Dictionary to store MSE for each lag
#     scoring_results = {}
#     score_results = {}
#
#     # corr = np.correlate(data_trace, design_matrix.iloc[:, 1], mode='full')
#     # # Number of samples in the original signals
#     # n = len(data_trace)  # Assumes both signals are the same length
#     # # Create an array of lag values. Lags range from -(n-1) to (n-1)
#     # corr_lags = np.arange(-n + 1, n)
#     # # Find the index of the maximum correlation value
#     # max_index = np.argmax(corr)
#     # corr_best_lag = corr_lags[max_index]
#     #
#     # fig, axs = plt.subplots(2, 1)
#     # axs[0].plot(data_trace, 'k')
#     # axs[1].plot(corr_lags, corr, marker='o')
#     # axs[1].set_xlabel('Lag')
#     # axs[1].set_ylabel('Correlation')
#     # axs[1].axvline(x=corr_best_lag, color='r', linestyle='--', label=f'Best lag = {corr_best_lag}')
#     # axs[1].legend()
#
#     # # Pad zeros to the beginning and the end for later time shifting  (15 % of data size)
#     # pad_start = int(data_trace.shape[0] * 0.5)
#     # pad_end = int(data_trace.shape[0] * 0.5)
#     #
#     # design_matrix_padded = pd.DataFrame(
#     #     np.pad(design_matrix.iloc[:, 1].to_numpy(), (pad_start, pad_end), mode='constant', constant_values=0)
#     # )
#     # design_matrix_padded = sm.add_constant(design_matrix_padded)
#     # design_matrix_padded.columns = design_matrix.columns
#     #
#     # data_trace_padded = pd.DataFrame(
#     #     np.pad(data_trace.to_numpy(), (pad_start, pad_end), mode='constant', constant_values=0)
#     # )
#
#     if center:
#         data_trace = data_trace - data_trace.mean()
#         design_matrix = design_matrix - design_matrix.mean()
#         design_matrix['const'] = 1
#
#     for _, lag in enumerate(possible_lags):
#         # Shift the regressors by the current lag.
#         # Note: The shift() function shifts the index by the given number of periods.
#         x_shifted = design_matrix.shift(lag)
#
#         # Combine y and shifted X into a single DataFrame and drop rows with NaNs.
#         df = pd.concat([data_trace, x_shifted], axis=1).dropna()
#         # df = pd.concat([data_trace, x_shifted], axis=1).fillna(0)
#
#         y_clean = df.iloc[:, 0]
#         x_clean = df.iloc[:, 1:]
#
#         # Compute LM-OLS Model
#         model = sm.OLS(y_clean, x_clean).fit()
#
#         # Compute Ridge Regression
#         # result_ridge = model.fit_regularized(alpha=0.01, L1_wt=0.0)
#         cf = model.params.iloc[1]
#
#         # Compute R2 Value
#         y_prediction = model.predict(design_matrix)  # Predicted calcium trace
#         r2 = r2_score(data_trace, y_prediction)  # Compute R² score
#         score = cf * r2
#         if score < 0:
#             score = 0
#         if r2 < 0:
#             score = 0
#         if cf < 0:
#             score = 0
#
#         # r2_results[lag] = r2
#         score_results[lag] = score
#
#         scoring_results[lag] = {
#             'r2': r2,
#             'cf': float(cf),
#             'score': float(score)
#         }
#         # if lag == 0:
#         #     axs[0].plot(x_clean.iloc[:, 1], 'k', lw=2)
#         # else:
#         #     axs[0].plot(x_clean.iloc[:, 1])
#         # plt.plot(x_clean.iloc[:, 1], 'r')
#         # plt.plot(y_clean, 'k')
#         # plt.title(f'lag: {lag}, R2={r2:.3f}, Score={score:.3f}')
#         # plt.show()
#
#     # Identify the lag that maximizes R2.
#     best_lag = max(score_results, key=score_results.get)
#     best_result = scoring_results[best_lag]
#     # axs[0].set_title(f'{design_matrix.columns[1]}-{tag}, Best Lag: {best_lag}, Score={best_result["score"]:.2f} (R2={best_result["r2"]:.2f})')
#     # plt.show()
#
#     return best_result, int(best_lag)
#



# def multiple_regression_model(ca_data, motor_events_spontaneous, motor_events_stimulus, visual_events, ca_sampling_rate, num_permutations=1000):
#     """
#     Performs multiple regression analysis to partition variance, adding a spontaneous activity regressor.
#
#     Parameters:
#     df (pd.DataFrame): Calcium traces (each column = one ROI).
#     motor_events (np.array): Binary motor event trace.
#     visual_events (np.array): Binary visual stimulus trace.
#     ca_sampling_rate (float): Calcium imaging sampling rate.
#     num_permutations (int): Number of permutations for statistical significance testing.
#
#     Returns:
#     pd.DataFrame: Contains full model R², unique and shared variance, and p-values for each ROI.
#     """
#
#     # Generate Calcium Impulse Response Function
#     _, cif = calcium_impulse_response(tau_rise=2, tau_decay=7, amplitude=1.0, sampling_rate=ca_sampling_rate,
#                                       threshold=1e-5, norm=True)
#
#     motor_events_all = motor_events_stimulus + motor_events_spontaneous
#     motor_events_all[motor_events_all > 1] = 1
#
#     # Generate calcium regressors
#     motor_spontaneous_regressor = create_regressors_from_binary(motor_events_spontaneous, cif, delta=True, norm=False)
#     motor_stimulus_regressor = create_regressors_from_binary(motor_events_stimulus, cif, delta=True, norm=False)
#     visual_regressor = create_regressors_from_binary(visual_events, cif, delta=True, norm=False)
#
#     # Get spontaneous Ca activity
#     ca_data_spontaneous_activity = get_spontaneous_activity(ca_data, ca_sampling_rate)
#     # ca_data_spontaneous_activity = get_spontaneous_activity2(
#     #     ca_data, motor_events_all, visual_events, ca_sampling_rate, sigma=0, width=10
#     # )
#
#     a = cutout_windowed_activity(ca_data, motor_events_spontaneous, motor_spontaneous_regressor, ca_sampling_rate
#                                  , width_left=2, width_right=10, stitch=False)
#
#     results = []
#     # Loop over each ROI (column in df)
#     for roi in ca_data.columns:
#         y_trace = ca_data[roi]  # Get calcium trace
#         spontaneous_component = ca_data_spontaneous_activity[roi]
#
#         # Create Design Matrix: Motor + Visual + Spontaneous with Intercept
#         fdm = np.column_stack((motor_spontaneous_regressor, motor_stimulus_regressor, visual_regressor, spontaneous_component))
#         fdm = sm.add_constant(fdm)
#         design_matrix_df = pd.DataFrame(fdm, columns=['const', 'motor_sp', 'motor_st', 'visual', 'spont'])
#
#         # Find best time shift
#         from_secs = 10
#         to_secs = 10
#         best_lag, r2_vals = find_best_lag(y_trace, design_matrix_df, from_secs, to_secs)
#         best_lag_secs = best_lag/ca_sampling_rate
#         # print('')
#         # print(f'Best lag: {best_lag_secs:.3f} s ({best_lag} samples): R²={r2_vals[best_lag]:.3f}')
#         # print(f'R² Diff. to 0 lag: {r2_vals[best_lag]-r2_vals[0]:.3f} (lag 0: R²={r2_vals[0]:.3f})')
#         # print('')
#
#         # Now use the best lag for shifting and compute the final model
#         # Fit Multiple Regression Model
#         shifted_design_matrix = design_matrix_df.shift(best_lag)
#
#         # Combine y and shifted X into a single DataFrame and drop rows with NaNs.
#         df = pd.concat([y_trace, shifted_design_matrix], axis=1).dropna()
#         y_trace = df[y_trace.name]
#         full_design_matrix_df = df[design_matrix_df.columns]
#         y_trace = y_trace.reset_index(drop=True)
#         full_design_matrix_df = full_design_matrix_df.reset_index(drop=True)
#
#         # Add Intercept
#         # full_design_matrix_df = sm.add_constant(full_design_matrix_df)
#
#         # Compute Full Model
#         model_full = sm.OLS(y_trace, full_design_matrix_df).fit()
#
#         # Compute Full Model R²
#         r2_full = model_full.rsquared
#
#         # Compute Semi-Partial R² for Motor Spontaneous
#         r2_no_spontaneous_motor = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['motor_sp'])).fit().rsquared
#         r2_motor_spontaneous_unique = max(r2_full - r2_no_spontaneous_motor, 0)
#
#         # Compute Semi-Partial R² for Motor Stimulus
#         r2_no_stimulus_motor = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['motor_st'])).fit().rsquared
#         r2_motor_stimulus_unique = max(r2_full - r2_no_stimulus_motor, 0)
#
#         # Compute Semi-Partial R² for Visual
#         r2_no_visual = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['visual'])).fit().rsquared
#         r2_visual_unique = max(r2_full - r2_no_visual, 0)
#
#         # Compute Semi-Partial R² for Spontaneous
#         r2_no_spontaneous = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['spont'])).fit().rsquared
#         r2_spontaneous_unique = max(r2_full - r2_no_spontaneous, 0)
#
#         # Motor Shared
#         r2_no_motor = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['motor_sp', 'motor_st'])).fit().rsquared
#         # Total contribution from both motor regressors
#         r2_motor_group = max(r2_full - r2_no_motor, 0)
#
#         # Shared contribution of motor regressors (overlap, should be around zero)
#         r2_shared_motor = max(r2_motor_group - (r2_motor_spontaneous_unique + r2_motor_stimulus_unique), 0)
#
#         # Motor Stimulus and Visual Stimulus
#         # Combine regressors that are not motor_stimulus or visual.
#         r2_no_motor_stim_visual = sm.OLS(y_trace, full_design_matrix_df.drop(columns=['motor_st', 'visual'])).fit().rsquared
#
#         # Total contribution of motor_stimulus and visual (unique + shared)
#         r2_motor_stim_visual_group = r2_full - r2_no_motor_stim_visual
#
#         # Unique contribution for visual:
#         shared_motorstim_visual = max(r2_motor_stim_visual_group - (r2_motor_stimulus_unique + r2_visual_unique), 0)
#
#         # Compute Shared Contribution
#         r2_shared = max(r2_full - r2_motor_spontaneous_unique - r2_motor_stimulus_unique - r2_visual_unique - r2_spontaneous_unique, 0)
#
#         # Compute Unexplained Variance
#         r2_unexplained = max(1 - r2_full, 0)
#
#         # Perform permutation test for significance
#         shuffled_scores = []
#         for _ in range(num_permutations):
#             shuffled_x = shuffle(full_design_matrix_df, random_state=None).reset_index(drop=True)
#             shuffled_model = sm.OLS(y_trace, shuffled_x).fit()
#             shuffled_scores.append(shuffled_model.rsquared)
#
#         # Compute p-value
#         p_value = np.sum(np.array(shuffled_scores) >= r2_full) / num_permutations
#
#         # Store results
#         results.append({
#             "ROI": roi,
#             "R2_Full_Model": float(r2_full),
#             "R2_Motor_Spontaneous_Unique": float(r2_motor_spontaneous_unique),
#             "R_Motor_Stimulus_Unique": float(r2_motor_stimulus_unique),
#             "R2_Motor_Group": float(r2_motor_group),
#             "R2_Shared_Motor": float(r2_shared_motor),  # <-- new entry for shared motor variance
#             "R2_Visual_Unique": float(r2_visual_unique),
#             "R2_Shared_MotorStim_Visual": float(shared_motorstim_visual),
#             "R2_Spontaneous_Unique": float(r2_spontaneous_unique),
#             "R2_Shared_all)": float(r2_shared),
#             "R2_Unexplained": float(r2_unexplained),
#             "P_Value": float(p_value)
#         })
#
#
#         # if best_lag < -8:
#         #     print('Y')
#         # else:
#         #     continue
#         #
#         # print('')
#         # print(f'Roi: {roi}')
#         # i = 0
#         # for k in results[0]:
#         #     i += 1
#         #     if i == 1:
#         #         continue
#         #     print(f'{k}: {results[0][k]:.3f}')
#         # print('')
#         #
#         # # Detect Peaks in Ca
#         # from scipy.signal import find_peaks
#         # peaks, _ = find_peaks(y_trace, height=np.mean(y_trace) + 2 * np.std(y_trace), distance=5)
#         #
#         # plt.plot(y_trace, 'k')
#         # # plt.plot(peaks, y_trace[peaks], 'rx')
#         # plt.plot(motor_spontaneous_regressor, 'r')
#         # plt.plot(full_design_matrix_df['motor_sp'], 'r--')
#         #
#         # plt.plot(motor_stimulus_regressor, 'b')
#         # plt.plot(full_design_matrix_df['motor_st'], 'b--')
#         #
#         # plt.plot(visual_regressor, 'g')
#         # plt.plot(full_design_matrix_df['visual'], 'g--')
#         #
#         # plt.plot(spontaneous_component, 'gray', ls='-')
#         # plt.plot(full_design_matrix_df['spont'], 'gray', ls='--')
#         #
#         # plt.show()
#         # embed()
#         # exit()
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#
#     return results_df


# def compute_nmf(ca_data):
#     from sklearn.decomposition import NMF
#
#     # ca_data_z = (ca_data - ca_data.mean()) / (ca_data.std())
#     ca_data_norm = (ca_data - ca_data.min()) / (ca_data.max() - ca_data.min())
#
#     # Apply Non-Negative Matrix Factorization (NMF)
#     n_components = 5# Adjust the number of components based on your dataset
#     nmf_model = NMF(n_components=n_components, init="nndsvd", random_state=42)
#     W = nmf_model.fit_transform(ca_data_norm.T)  # Neurons x Components
#     H = nmf_model.components_  # Components x Time
#
#     # Convert W matrix (Neuron Loadings) to DataFrame
#     W_df = pd.DataFrame(W, index=ca_data.columns, columns=[f"Component {i + 1}" for i in range(n_components)])
#     H_df = pd.DataFrame(H.T, columns=[f"Component {i + 1}" for i in range(n_components)])
#
#     return W_df, H_df
#
#
# def compute_clustering(W_df, H_df, ca_data, ca_sampling_rate, stimulus_trace):
#     import seaborn as sns
#     from sklearn.cluster import AgglomerativeClustering
#     from sklearn.decomposition import PCA
#     import scipy.cluster.hierarchy as sch
#
#     # Number of clusters (adjust based on your data)
#     n_clusters = 5
#     n_components = H_df.shape[1]
#
#     # Perform Agglomerative Clustering
#     cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
#     # cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='correlation', linkage='average')
#
#     W_df["Cluster"] = cluster_model.fit_predict(W_df)
#
#     # Sort neurons by cluster for visualization
#     W_df_sorted = W_df.sort_values("Cluster")
#     ca_data_transpose = ca_data.T
#     ca_data_transpose['Cluster'] = W_df['Cluster']
#     ca_data_sorted = ca_data_transpose.sort_values('Cluster')
#
#     # Plot the component time series (H matrix)
#     ca_time = np.linspace(0, ca_data.shape[0] / ca_sampling_rate, ca_data.shape[0])
#     fig, axs = plt.subplots(n_components+1, 1)
#     for i in range(n_components+1):
#         if i == 0:
#             axs[i].plot(ca_time, stimulus_trace, 'b', label='Stimulus')
#         else:
#             axs[i].plot(H_df.iloc[:, i-1], 'k', label=f"Component {i}")
#         axs[i].legend()
#
#     axs[-1].set_xlabel("Time")
#     axs[-1].set_ylabel("Component Activity")
#
#     # Plot heatmap of NMF loadings with clustering
#     sns.clustermap(W_df_sorted.drop(columns=["Cluster"]), cmap="coolwarm", standard_scale=1, row_cluster=False)
#     plt.title("Neuron Functional Clustering (NMF Loadings)")
#
#     # Hierarchical dendrogram
#     plt.figure(figsize=(10, 5))
#     sch.dendrogram(sch.linkage(W_df.drop(columns=["Cluster"]), method="ward"))
#     plt.title("Hierarchical Clustering Dendrogram")
#     plt.xlabel("Neuron Index")
#     plt.ylabel("Distance")
#
#     # PCA Projection for Cluster Visualization
#     pca = PCA(n_components=2)
#     W_pca = pca.fit_transform(W_df.drop(columns=["Cluster"]))
#
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(W_pca[:, 0], W_pca[:, 1], c=W_df["Cluster"], cmap="tab10", alpha=0.7)
#     plt.colorbar(scatter, label="Cluster")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.title("Neuron Clusters (PCA Projection)")
#
#     plt.show()
#
#     # HEATMAP
#     fig, axs = plt.subplots()
#     v_min = 0
#     v_max = 3
#     number_of_rois = ca_data_sorted.shape[0]
#     cmap = 'afmhot'
#     axs.imshow(
#         ca_data_sorted, aspect='auto', extent=(ca_time[0], ca_time[-1], 0, number_of_rois),
#         cmap=cmap, origin='lower', vmin=v_min, vmax=v_max
#     )
#     axs.set_xlim([0, ca_time[-1]])
#     axs.set_ylim([0.5, number_of_rois+0.5])
#     axs.set_xlabel('Time [s]')
#     axs.set_ylabel(f'ROIs (n={number_of_rois})')
#     plt.show()
#
#     embed()
#     exit()
#
