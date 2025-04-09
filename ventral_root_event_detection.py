import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from utils import save_dict_as_hdf5, moving_average_filter, z_transform, filter_high_pass
from IPython import embed
"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'ventral_root_event_detection.py'

Detect motor events (swim bouts) in the ventral root recording of selected sweeps.
It will compute the envelope of the ventral roo trace and then detect onset and offset of motor events.
Additionally, it will compute the duration and power of motor events. If the stimulus protocol is provided it will also
classify motor events as spontaneous or stimulus associated ("check_stimulus = True").

By setting "detection_validation = False" you can visually check the event detection for a specific sweep.

Result: 
    - /ventral_root/selected_sweeps_ventral_root_events.h5
    - /ventral_root/selected_sweeps_ventral_root_all_binaries.h5
    - /ventral_root/selected_sweeps_ventral_root_binaries.csv
    - /ventral_root/selected_sweeps_ventral_root_binaries_spontaneous.csv
    - /ventral_root/selected_sweeps_ventral_root_binaries_stimulus.csv
...

Requirements:
    - /ventral_root/selected_sweeps_aligned_vr_recordings.h5'
    - /meta_data/sampling_rate.csv'
    - /data/df_f_data.csv'
    - /stimulus_traces/stimulus_traces.csv
    - /protocols/

You can adjust the detection by changing the following parameters:
    settings = {
        'vr_sampling_rate': 10000,  # VR Sampling Rate in Hz
        'vr_trace_low_pass_cutoff': 500,  # Low Pass Filter Cut Off for Vr Trace (remove slow oscillations)
        'detection_threshold': 1.5,  # in SD
        'duration_threshold': 2,  # in seconds. Events longer than that will be ignored
        'min_duration_threshold': 0.1,  # in seconds. Events shorter than that will be ignored
        'envelope_low_pass_cut_off': 5,  # Low Pass Filter Cut Off used in envelope computation in Hz (5)
        'ca_sampling_rate': ca_sampling_rate,
        'ignore_first_seconds': 5,  # ignore the first seconds of the vr recording
        'ignore_last_seconds': 2,  # ignore the last seconds of the vr recording
        'remove_high_values': 20,  # in SD
        'env_moving_average_window': 20,  # in samples: The Window for the moving average filter for the envelope signal
        'min_time_merging': 0,  # minimum time in seconds. If two events are too close merge them. Ignore if set to 0
        'peak_prominence': 1,  # scipy find_peaks prominence. Is used to identify real signals and noise.
        'before_stimulus': 2,  # buffer before stimulus onset for classifying swim events in seconds
        'after_stimulus': 5,  # buffer after stimulus onset for classifying swim events in seconds
        'check_symmetry': 'energy',  # Check Symmetry Method (None, stats, freq, energy
        'check_symmetry_parameter': 0.4,  # Depends on Method
    }

Nils Brehm  -  2025
"""


def merge_events(onset_times, offset_times, threshold):
    """
    Merges events that are close to each other within a specified threshold.

    Parameters:
    - onset_times: List of onset times of events.
    - offset_times: List of offset times of events.
    - threshold: Minimum time gap to consider merging two events.

    Returns:
    - merged_onsets: List of merged onset times.
    - merged_offsets: List of merged offset times.
    """
    if len(onset_times) != len(offset_times):
        raise ValueError("onset_times and offset_times must have the same length.")

    # Initialize merged lists
    merged_onsets = []
    merged_offsets = []

    # Start with the first event
    current_onset = onset_times[0]
    current_offset = offset_times[0]

    for i in range(1, len(onset_times)):
        # If the next event is within the threshold, merge it
        if onset_times[i] - current_offset <= threshold:
            # Extend the current event
            current_offset = max(current_offset, offset_times[i])
        else:
            # Finalize the current event and move to the next
            merged_onsets.append(current_onset)
            merged_offsets.append(current_offset)
            current_onset = onset_times[i]
            current_offset = offset_times[i]

    # Append the last event
    merged_onsets.append(current_onset)
    merged_offsets.append(current_offset)

    return np.array(merged_onsets), np.array(merged_offsets)

def test_plot(vr, ca_time_axis, vr_time, event_activity, vr_binary_df, vr_env, stimulus, detection_th):
    def on_key(event):
        """Event handler for key presses to navigate the x-axis."""
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        step = (xlim[1] - xlim[0]) * 0.1  # Move 10% of the current x-axis range
        y_step = (ylim[1] - ylim[0]) * 0.1

        if event.key == 'd':  # Move to the right
            ax.set_xlim(xlim[0] + step, xlim[1] + step)
        elif event.key == 'a':  # Move to the left
            ax.set_xlim(xlim[0] - step, xlim[1] - step)

        elif event.key == 'up' or event.key == '.':  # Zoom in
            ax.set_xlim(xlim[0] + step, xlim[1] - step)
        elif event.key == 'down' or event.key == ',':  # Zoom out
            ax.set_xlim(xlim[0] - step, xlim[1] + step)

        elif event.key == 'x':  # Y Axis Zoom in
            ax.set_ylim(ylim[0] + y_step, ylim[1] - y_step)
        elif event.key == 'y':  # Y Axis Zoom out
            ax.set_ylim(ylim[0] - y_step, ylim[1] + y_step)

        plt.draw()

    print('')
    print('==== Plot Navigation =============')
    print('d or .: move x-axis to the right')
    print('a or ,: move x-axis to the left')
    print('arrow key up: zoom in x-axis')
    print('arrow key down: zoom out x-axis')
    print('x: zoom in y-axis')
    print('y: zoom out y-axis')
    print('==================================')
    print('')

    # Create the plot
    fig, ax = plt.subplots()

    # Ventral Root Recording Trace
    ax.plot(vr_time, vr, 'grey', label='VRR')

    # Detection Threshold
    ax.plot(vr_time, np.zeros_like(vr_time) + detection_th, 'r--', label='Detection Threshold')

    # VR Envelope
    ax.plot(vr_time, vr_env, 'tab:orange', label='VRR Envelope')

    # VR Binary down-sampled to Ca Imaging resolution
    ax.plot(ca_time_axis, vr_binary_df * np.max(vr), 'tab:green', label='VRR Binary (Imaging Time)')

    # VR Event onset time (in VR recording resolution)
    ax.plot(event_activity['start_time'], np.zeros_like(event_activity['start_time']) + np.max(vr), 'bx',
            label='VRR Event Start (VRR Time)')

    # Stop of Ca Imaging Recording (mostly VR recording is a bit longer)
    ax.plot([ca_time_axis.max(), ca_time_axis.max()], [np.min(vr), np.max(vr_env)], 'r:',
            label='End of Imaging Recording')

    # Plot Stimulus
    if stimulus is not None:
        ax.plot(ca_time_axis, stimulus, 'blue', lw=2, label='Stimulus')
        stimulus_events = event_activity[event_activity['stimulus'] != 'spontaneous']['start_time']
        ax.plot(
            stimulus_events,
            (np.zeros_like(stimulus_events) + np.max(vr)),
            'rx'
        )

    # Indicate the Ca Imaging Resolution (Sampling rate) which is much lower than VR sampling rate
    for k in range(20):
        if k == 0:
            ax.plot(ca_time_axis, np.ones_like(ca_time_axis) + k, 'g.', markersize=0.25,
                    label='Imaging Sampling Resolution', alpha=0.5)
        else:
            ax.plot(ca_time_axis, np.ones_like(ca_time_axis) + k, 'g.', markersize=0.25, alpha=0.5)

    ax.legend(loc='lower left')

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def envelope(data, rate, freq):
    # Low pass filter the absolute values of the signal in both forward and reverse directions,
    # resulting in zero-phase filtering.
    sos = sig.butter(2, freq, 'lowpass', fs=rate, output='sos')
    lp_abs = sig.sosfiltfilt(sos, np.abs(data))

    # [square_root(2) * low_pass(absolute(data))] ** 2
    env = (np.sqrt(2) * lp_abs) ** 2
    return env


def find_closest_times(time_points, time_axis):
    """
    Finds the closest points in the time axis for each time point.

    Parameters:
    - time_points: Array of time points to match (list or NumPy array of floats).
    - time_axis: The lower-resolution time axis (list or NumPy array of floats).

    Returns:
    - closest_times: List of closest times for each time point.
    """
    closest_times = []
    indices = []
    for time_point in time_points:
        # Find the closest time in the time_axis
        closest_time = min(time_axis, key=lambda x: abs(x - time_point))
        index = int(np.where(time_axis == closest_time)[0][0])
        closest_times.append(closest_time)
        indices.append(index)

    return np.array(closest_times), np.array(indices)


def find_closest_past_times(time_points, time_axis):
    """
    Finds the closest points in the time axis that are in the past for each time point.

    Parameters:
    - time_points: Array of time points to match (list or NumPy array of floats).
    - time_axis: The lower-resolution time axis (list or NumPy array of floats).

    Returns:
    - closest_times: List of closest past times for each time point.
    """
    closest_times = []
    indices = []
    for time_point in time_points:
        # Filter time_axis to include only points >= time_point
        past_times = [t for t in time_axis if t >= time_point]
        if past_times:
            closest_time_point = min(past_times)
            index = int(np.where(time_axis == closest_time_point)[0][0])
            closest_times.append(closest_time_point)  # Min is the closest in the past
            indices.append(index)
        else:
            closest_times.append(None)  # No past time available

    return np.array(closest_times), np.array(indices)


def ignore_start_and_end(data, time_axis, first_seconds, last_seconds):
    # Ignore first secs of recording (= set them to zero)
    data[time_axis <= first_seconds] = 0
    data_max_time = time_axis.max()
    data[data >= data_max_time - last_seconds] = 0

    return data


def replace_high_values(data, th_remove):
    # Replace high values from the VR Trace with mean values
    data[data > (np.mean(data) + th_remove * np.std(data))] = np.mean(data)
    data[data < (np.mean(data) + th_remove * np.std(data))*-1] = np.mean(data)
    return data


def find_onsets_offsets(binary, vr_time):
    # Find onsets and offsets of ventral root activity
    onsets_offsets = np.diff(binary, append=0)

    onset_idx = np.where(onsets_offsets > 0)[0]
    offset_idx = np.where(onsets_offsets < 0)[0]

    if offset_idx.shape[0] > onset_idx.shape[0]:
        print('WARNING: More Offsets than Onsets Found!')
        offset_idx = offset_idx[:onset_idx.shape[0]].copy()

    if offset_idx.shape[0] < onset_idx.shape[0]:
        print('WHAAAT ... More onset than offset????')
        embed()
        exit()
    onset_times = vr_time[onset_idx]
    offset_times = vr_time[offset_idx]

    return onset_times, offset_times, onset_idx, offset_idx


# def remove_unsymmetrical_signals(data, th_below, onset_idx, offset_idx, onset_times, offset_times):
#     # Remove Signals that are not symmetrical (only have positive values)
#     d_below = []
#     d_above = []
#     for on, off in zip(onset_idx, offset_idx):
#         d_below.append(np.sum(data[on:off] < -th_below))
#         d_above.append(np.sum(data[on:off] > th_below))
#
#     idx_below = (np.array(d_below) <= 1)
#     idx_above = (np.array(d_above) <= 1)
#     new_onset_times = onset_times[np.invert(idx_above+idx_below)]
#     new_offset_times = offset_times[np.invert(idx_above+idx_below)]
#     return new_onset_times, new_offset_times

def filter_symmetric_signals_energy_with_smoothing(data, onset_idx, offset_idx, onset_times, offset_times,
                                                   energy_ratio_thresh=0.1, window_length=51, polyorder=3):
    """
    Filters signals based on symmetry using energy comparison, with a smoothing step.

    Parameters:
    - data: ndarray
        The array containing the signal data.
    - onset_idx: list or ndarray
        Indices marking the start of signals in the data.
    - offset_idx: list or ndarray
        Indices marking the end of signals in the data.
    - onset_times: list or ndarray
        Timestamps corresponding to the start of each signal.
    - offset_times: list or ndarray
        Timestamps corresponding to the end of each signal.
    - energy_ratio_thresh: float, optional (default=0.1)
        Threshold for the relative difference between positive and negative energy.
        Signals with a difference below this threshold are considered symmetrical.
    - window_length: int, optional (default=51)
        The length of the smoothing window for Savitzky-Golay filter (must be odd).
    - polyorder: int, optional (default=3)
        The order of the polynomial used for the Savitzky-Golay filter.

    Returns:
    - new_onset_times: ndarray
        The onset times of symmetrical signals.
    - new_offset_times: ndarray
        The offset times of symmetrical signals.
    """
    from scipy.signal import savgol_filter
    # Smooth the entire signal using Savitzky-Golay filter
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    # List to store the indices of valid (symmetrical) signals
    valid_indices = []

    # Loop through each signal segment
    for i, (on, off) in enumerate(zip(onset_idx, offset_idx)):
        segment = smoothed_data[on:off]

        # Calculate positive and negative energy
        positive_energy = np.sum(segment[segment > 0] ** 2)
        negative_energy = np.sum(segment[segment < 0] ** 2)

        # Compute energy ratio (relative difference between positive and negative energy)
        total_energy = positive_energy + negative_energy + 1e-10  # Avoid division by zero
        energy_ratio = abs(positive_energy - abs(negative_energy)) / total_energy

        # Check if the energy ratio meets the threshold
        if energy_ratio <= energy_ratio_thresh:
            valid_indices.append(i)

    # Filter onset and offset times based on valid indices
    new_onset_times = np.array(onset_times)[valid_indices]
    new_offset_times = np.array(offset_times)[valid_indices]

    return new_onset_times, new_offset_times


def envelope_roc(env, sampling_rate, onset_times, offset_times):
    """
    """
    # Convert onset and offset times to indices
    onset_idx = (np.array(onset_times) * sampling_rate).astype(int)
    offset_idx = (np.array(offset_times) * sampling_rate).astype(int)

    # Initialize a list to store the power of each segment
    segment_powers = []
    segment_integral = []

    # Loop through each signal segment
    for i, (on, off) in enumerate(zip(onset_idx, offset_idx)):
        segment = env[on:off]

        # Compute integral of the signal (envelope is alsway positive)
        env_integral = np.sum(segment)
        power = env_integral / len(segment)

        # Append the power to the results list
        segment_powers.append(power)
        segment_integral.append(env_integral)

    return np.array(segment_powers), np.array(segment_integral)


def event_power(data, sampling_rate, onset_times, offset_times, window_length=51, polyorder=3):
    """
    Computes the power of signal segments based on onset and offset times.

    Parameters:
    - data: ndarray
        The array containing the signal data.
    - sampling_rate: float
        The sampling rate of the signal (in Hz).
    - window_length: int, optional (default=51)
        The length of the smoothing window for Savitzky-Golay filter (must be odd).
    - polyorder: int, optional (default=3)
        The order of the polynomial used for the Savitzky-Golay filter.
    - onset_times: list or ndarray
        The start times (in seconds) of the segments of interest.
    - offset_times: list or ndarray
        The end times (in seconds) of the segments of interest.

    Returns:
    - segment_powers: list
        The power of each segment.
    """
    from scipy.signal import savgol_filter

    # Smooth the entire signal using Savitzky-Golay filter
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)

    # Convert onset and offset times to indices
    onset_idx = (np.array(onset_times) * sampling_rate).astype(int)
    offset_idx = (np.array(offset_times) * sampling_rate).astype(int)

    # Initialize a list to store the power of each segment
    segment_powers = []
    segment_energies = []

    # Loop through each signal segment
    for i, (on, off) in enumerate(zip(onset_idx, offset_idx)):
        segment = smoothed_data[on:off]

        # Compute power of the signal (mean of squared values)
        power = np.mean(segment ** 2)

        # Compute energy of the signal (sum of squared values)
        energy = np.sum(segment ** 2)

        # Append the power to the results list
        segment_powers.append(power)
        segment_energies.append(energy)

    return np.array(segment_powers), np.array(segment_energies)


def filter_symmetric_signals_frequency(data, onset_idx, offset_idx, onset_times, offset_times, power_ratio_thresh=0.1):
    """
    Filters signals based on symmetry using frequency-domain analysis.

    Parameters:
    - data: ndarray
        The array containing the signal data.
    - onset_idx: list or ndarray
        Indices marking the start of signals in the data.
    - offset_idx: list or ndarray
        Indices marking the end of signals in the data.
    - onset_times: list or ndarray
        Timestamps corresponding to the start of each signal.
    - offset_times: list or ndarray
        Timestamps corresponding to the end of each signal.
    - power_ratio_thresh: float, optional (default=0.1)
        Threshold for the ratio of low-frequency to high-frequency power.
        Symmetrical signals typically have a balanced power spectrum.

    Returns:
    - new_onset_times: ndarray
        The onset times of symmetrical signals.
    - new_offset_times: ndarray
        The offset times of symmetrical signals.
    """
    # List to store the indices of valid (symmetrical) signals
    valid_indices = []

    # Loop through each signal segment
    for i, (on, off) in enumerate(zip(onset_idx, offset_idx)):
        segment = data[on:off]

        # Compute the Fast Fourier Transform (FFT)
        fft_data = np.fft.fft(segment)
        power_spectrum = np.abs(fft_data) ** 2

        # Separate low-frequency and high-frequency components
        midpoint = len(power_spectrum) // 2
        low_frequency_power = np.sum(power_spectrum[:midpoint])  # Sum of low frequencies
        high_frequency_power = np.sum(power_spectrum[midpoint:])  # Sum of high frequencies

        # Calculate the power ratio
        power_ratio = low_frequency_power / (high_frequency_power + 1e-10)  # Avoid division by zero

        # Check if the power ratio indicates symmetry
        if abs(power_ratio - 1) <= power_ratio_thresh:
            valid_indices.append(i)

    # Filter onset and offset times based on valid indices
    new_onset_times = np.array(onset_times)[valid_indices]
    new_offset_times = np.array(offset_times)[valid_indices]

    return new_onset_times, new_offset_times


def filter_symmetric_signals(data, onset_idx, offset_idx, onset_times, offset_times, skew_thresh=0.5, kurt_thresh=10):
    """
    Filters signals based on symmetry using skewness and kurtosis.

    Parameters:
    - data: ndarray
        The array containing the signal data.
    - onset_idx: list or ndarray
        Indices marking the start of signals in the data.
    - offset_idx: list or ndarray
        Indices marking the end of signals in the data.
    - onset_times: list or ndarray
        Timestamps corresponding to the start of each signal.
    - offset_times: list or ndarray
        Timestamps corresponding to the end of each signal.
    - skew_thresh: float, optional (default=0.5)
        Threshold for skewness. Signals with absolute skewness below this value are considered symmetrical.
    - kurt_thresh: float, optional (default=10)
        Threshold for kurtosis. Signals with kurtosis below this value are considered valid.

    Returns:
    - new_onset_times: ndarray
        The onset times of symmetrical signals.
    - new_offset_times: ndarray
        The offset times of symmetrical signals.
    """
    from scipy.stats import skew, kurtosis

    # Lists to store the indices of valid (symmetrical) signals
    valid_indices = []

    # Loop through each signal segment
    for i, (on, off) in enumerate(zip(onset_idx, offset_idx)):
        segment = data[on:off]

        # Compute skewness and kurtosis of the segment
        segment_skewness = skew(segment)
        segment_kurtosis = kurtosis(segment)

        # Check if the segment meets the skewness and kurtosis criteria
        if abs(segment_skewness) <= skew_thresh and segment_kurtosis <= kurt_thresh:
            valid_indices.append(i)

    # Filter onset and offset times based on valid indices
    new_onset_times = np.array(onset_times)[valid_indices]
    new_offset_times = np.array(offset_times)[valid_indices]

    return new_onset_times, new_offset_times


def remove_unsymmetrical_signals(data, th_below, onset_idx, offset_idx, onset_times, offset_times):
    """
    Remove unsymmetrical signals from the data.

    This function filters out signals that are not symmetrical based on a threshold.
    A symmetrical signal is assumed to have both positive and negative values exceeding the specified threshold.

    Parameters:
    - data: ndarray
        The array containing the signal data.
    - th_below: float
        The threshold value to determine symmetry. Signals must have both positive and negative values exceeding this threshold.
    - onset_idx: list or ndarray
        Indices in the data array where signals start.
    - offset_idx: list or ndarray
        Indices in the data array where signals end.
    - onset_times: list or ndarray
        Timestamps corresponding to the start of each signal.
    - offset_times: list or ndarray
        Timestamps corresponding to the end of each signal.

    Returns:
    - new_onset_times: ndarray
        The onset times of the symmetrical signals.
    - new_offset_times: ndarray
        The offset times of the symmetrical signals.
    """
    # Initialize lists to store counts of negative and positive values for each signal segment
    d_below = []  # To count how many values are below -th_below
    d_above = []  # To count how many values are above th_below

    # Iterate over the start (onset) and end (offset) indices of each signal
    for on, off in zip(onset_idx, offset_idx):
        # Count values below and above the threshold within the signal segment
        d_below.append(np.sum(data[on:off] < -th_below))  # Count values less than -th_below
        d_above.append(np.sum(data[on:off] > th_below))   # Count values greater than th_below

    # Determine signals with insufficient negative or positive values
    idx_below = (np.array(d_below) <= 1)  # Signals with <= 1 negative value below -th_below
    idx_above = (np.array(d_above) <= 1)  # Signals with <= 1 positive value above th_below

    # Combine conditions: a signal is unsymmetrical if it meets either idx_below or idx_above
    symmetrical_signals_mask = np.invert(idx_above + idx_below)

    # Filter onset and offset times to retain only symmetrical signals
    new_onset_times = onset_times[symmetrical_signals_mask]
    new_offset_times = offset_times[symmetrical_signals_mask]

    # Return the filtered onset and offset times
    return new_onset_times, new_offset_times


def remove_too_long_events(onset_times, offset_times, duration_th):
    event_duration = offset_times - onset_times
    idx_remove = event_duration > duration_th
    new_onset_times = onset_times[np.invert(idx_remove)]
    new_offset_times = offset_times[np.invert(idx_remove)]
    return new_onset_times, new_offset_times


def remove_too_short_events(onset_times, offset_times, duration_th):
    event_duration = offset_times - onset_times
    idx_remove = event_duration <= duration_th
    new_onset_times = onset_times[np.invert(idx_remove)]
    new_offset_times = offset_times[np.invert(idx_remove)]
    return new_onset_times, new_offset_times


def check_if_data_inside_boundaries(onset_times, offset_times, ca_time_axis):
    # Check Onset Times
    idx = onset_times < ca_time_axis.max()
    onset_times = onset_times[idx].copy()
    offset_times = offset_times[idx].copy()

    # Check Offset Times
    idx = offset_times < ca_time_axis.max()
    onset_times = onset_times[idx].copy()
    offset_times = offset_times[idx].copy()

    return onset_times, offset_times


def create_binary_from_events(onset_idx_ca_rec, offset_idx_ca_rec, ca_time_axis):
    # Create a Binary Trace from events
    event_binary = np.zeros_like(ca_time_axis)
    for event_s, event_e in zip(onset_idx_ca_rec, offset_idx_ca_rec):
        event_binary[event_s:event_e+1] = 1
    return event_binary


def validate_signal_symmetry(vr_trace, onset_idx, offset_idx, onset_times, offset_times, parameter, method='stats'):
    """
    # Comparison of Symmetry Detection Methods

    | **Method**               | **Efficiency** | **Robustness** | **Use Case**                        |
    |--------------------------|----------------|----------------|-------------------------------------|
    | Statistical Measures      | High           | Moderate       | Simple and quick symmetry check    |
    | Frequency-Domain Analysis | Moderate       | High           | Complex signals, noise resilience  |
    | Peak Analysis             | Moderate       | High           | Signals with distinct peaks        |
    | Smoothing/Filtering       | High           | High           | Noisy signals                      |
    | Symmetry Index            | High           | Moderate       | Custom analysis                    |
    | Machine Learning          | Low            | High           | Complex or large datasets          |
    | Energy Comparison         | High           | Moderate       | Amplitude-focused analysis         |
    | Cross-Correlation         | Moderate       | High           | Periodic or repeating signals      |
    """

    if method == 'stats':
        # Symmetry validation with Statistical Measures
        # good values: skew_thresh=2.0, kurt_thresh=30
        new_onset_times, new_offset_times = filter_symmetric_signals(
            vr_trace, onset_idx, offset_idx, onset_times, offset_times, skew_thresh=parameter[0], kurt_thresh=parameter[1]
        )
    elif method =='freq':
        # Symmetry validation with Frequency-Domain Analysis
        # good value: power_ratio_thresh=0.05
        new_onset_times, new_offset_times = filter_symmetric_signals_frequency(
            vr_trace, onset_idx, offset_idx, onset_times, offset_times, power_ratio_thresh=parameter
        )
    elif method == 'energy':
        # Symmetry validation with Smoothing and Energy Comparison (Amplitude-focused analysis)
        # good value: energy_ratio_thresh=0.5,
        new_onset_times, new_offset_times = filter_symmetric_signals_energy_with_smoothing(
            vr_trace, onset_idx, offset_idx, onset_times, offset_times,
            energy_ratio_thresh=parameter, window_length=31, polyorder=3
        )
    else:
        print('Symmetry Detection Method Not Found!')
        embed()
        exit()

    return new_onset_times, new_offset_times


def check_stimulus_overlap2(stimulus_protocol, onset_times, stimulus_earlier, stimulus_later):
    starts = stimulus_protocol[stimulus_protocol[3] == 'start'][4].to_list()[1:]
    stops = stimulus_protocol[stimulus_protocol[3] == 'stop'][4].to_list()

    starts.append(stimulus_protocol[stimulus_protocol[3] == 'appears'][4].item())
    stops.append(stimulus_protocol[stimulus_protocol[3] == 'appears'][4].item() + 5)

    # Add Grating Disappears (+ 5 seconds)
    starts.append(stimulus_protocol[stimulus_protocol[3] == 'disappears'][4].item())
    stops.append(stimulus_protocol[stimulus_protocol[3] == 'disappears'][4].item() + 5)

    # Add Bright Flash (+ 5 seconds)
    starts.append(stimulus_protocol.iloc[-2, 4].item())
    stops.append(stimulus_protocol.iloc[-2, 4].item() + 5)

    # Add Dark Flash (+ 5 seconds)
    starts.append(stimulus_protocol.iloc[-1, 4].item())
    stops.append(stimulus_protocol.iloc[-1, 4].item() + 5)

    # Combine start_times and stop_times into a list of ranges
    time_ranges = list(zip(starts, stops))
    # Find onset times within any of the time ranges
    results = []
    for onset_time in onset_times:
        in_range = False
        for start, stop in time_ranges:
            start_time = start - stimulus_earlier  # go 2 second earlier
            stop_time = stop + stimulus_later  # go 2 seconds later
            if start_time <= onset_time <= stop_time:
                in_range = True
                break  # Exit the loop as soon as we find a match
        results.append((onset_time, in_range))

    return np.array(results)


def collect_stimulus_times(stimulus_protocol):
    # Collect Stimulus onset and offset times
    stimulus_onsets = dict()
    stimulus_offsets = dict()

    # Moving Target
    idx = stimulus_protocol[1] == 'moving_target'
    stimulus_onsets['moving_target_01'] = stimulus_protocol[idx].iloc[0, 4]
    stimulus_offsets['moving_target_01'] = stimulus_protocol[idx].iloc[1, 4]
    stimulus_onsets['moving_target_02'] = stimulus_protocol[idx].iloc[2, 4]
    stimulus_offsets['moving_target_02'] = stimulus_protocol[idx].iloc[3, 4]

    # Grating Appears
    idx = stimulus_protocol[3] == 'appears'
    stimulus_onsets['grating_appears'] = stimulus_protocol[idx].iloc[0, 4]
    stimulus_offsets['grating_appears'] = stimulus_protocol[idx].iloc[0, 4] + 5

    # Grating Disappears
    idx = stimulus_protocol[3] == 'disappears'
    stimulus_onsets['grating_disappears'] = stimulus_protocol[idx].iloc[0, 4]
    stimulus_offsets['grating_disappears'] = stimulus_protocol[idx].iloc[0, 4] + 5

    # Grating 0
    idx = (stimulus_protocol[2] == '0') * (stimulus_protocol[3] == 'start')
    stimulus_onsets['grating_0'] = stimulus_protocol[idx].iloc[0, 4]
    idx = (stimulus_protocol[2] == '0') * (stimulus_protocol[3] == 'stop')
    stimulus_offsets['grating_0'] = stimulus_protocol[idx].iloc[0, 4]

    # Grating 180
    idx = (stimulus_protocol[2] == '180') * (stimulus_protocol[3] == 'start')
    stimulus_onsets['grating_180'] = stimulus_protocol[idx].iloc[0, 4]
    idx = (stimulus_protocol[2] == '180') * (stimulus_protocol[3] == 'stop')
    stimulus_offsets['grating_180'] = stimulus_protocol[idx].iloc[0, 4]

    # Bright Loom
    idx = (stimulus_protocol[2] == 'bright') * (stimulus_protocol[3] == 'start')
    stimulus_onsets['bright_loom'] = stimulus_protocol[idx].iloc[0, 4]
    idx = (stimulus_protocol[2] == 'bright') * (stimulus_protocol[3] == 'stop')
    stimulus_offsets['bright_loom'] = stimulus_protocol[idx].iloc[0, 4]

    # Dark Loom
    idx = (stimulus_protocol[2] == 'dark') * (stimulus_protocol[3] == 'start')
    stimulus_onsets['dark_loom'] = stimulus_protocol[idx].iloc[0, 4]
    idx = (stimulus_protocol[2] == 'dark') * (stimulus_protocol[3] == 'stop')
    stimulus_offsets['dark_loom'] = stimulus_protocol[idx].iloc[0, 4]

    # Bright Flash
    idx = (stimulus_protocol[2] == 'bright') * (stimulus_protocol[3] == 'ON')
    stimulus_onsets['bright_flash'] = stimulus_protocol[idx].iloc[0, 4]
    stimulus_offsets['bright_flash'] = stimulus_protocol[idx].iloc[0, 4] + 5

    # Dark Flash
    idx = (stimulus_protocol[2] == 'dark') * (stimulus_protocol[3] == 'ON')
    stimulus_onsets['dark_flash'] = stimulus_protocol[idx].iloc[0, 4]
    stimulus_offsets['dark_flash'] = stimulus_protocol[idx].iloc[0, 4] + 5

    return stimulus_onsets, stimulus_offsets


def check_stimulus_overlap(stimulus_protocol, motor_onset_times, settings):
    # Collect stimulus onset and offset times
    stimulus_onsets, stimulus_offsets = collect_stimulus_times(stimulus_protocol)

    # Loop over motor events onset times
    results = []
    for on_t in motor_onset_times:
        in_range = False
        # Loop over stimulus onsets and offsets
        for s_type in stimulus_onsets:
            s_on = stimulus_onsets[s_type] - settings[s_type][0]  # go x second earlier
            s_off = stimulus_offsets[s_type] + settings[s_type][1]  # go y seconds later
            if s_on <= on_t <= s_off:
                # Motor event falls into this stimulus window
                results.append((on_t, s_type))
                in_range = True
                break  # Exit the loop as soon as we find a match
        if not in_range:
            results.append((on_t, 'spontaneous'))

    return np.array(results)


def detect_peaks(signal, height=None, distance=None, width=None, prominence=None, threshold=None):
    """
    Detect peaks in a signal with adjustable criteria.

    Parameters:
        signal (array-like): The input signal (1D array).
        height (float or tuple, optional): Minimum height of peaks. Can be a scalar or (min, max).
        distance (float, optional): Minimum horizontal distance between peaks (in samples).
        width (float or tuple, optional): Minimum width of peaks at half-prominence. Can be a scalar or (min, max).
        prominence (float or tuple, optional): Required prominence of peaks. Can be a scalar or (min, max).
        threshold (float or tuple, optional): Required threshold of peaks, the vertical distance to its neighboring samples.

    Returns:
        dict: A dictionary with peak indices and properties.
            - 'peaks': Indices of the detected peaks.
            - 'properties': Properties of the detected peaks (e.g., widths, prominences).
    """
    from scipy.signal import find_peaks, peak_widths

    # Detect peaks based on criteria
    peaks, properties = find_peaks(
        signal, height=height, distance=distance, width=width, prominence=prominence, threshold=threshold
    )

    # Extract width properties if peaks were found
    if len(peaks) > 0:
        widths_result = peak_widths(signal, peaks, rel_height=0.5)
        properties['widths'] = widths_result[0]
        properties['width_heights'] = widths_result[1]
        properties['left_ips'] = widths_result[2]
        properties['right_ips'] = widths_result[3]

    return {'peaks': peaks, 'properties': properties}


def check_peaks(onset_times, offset_times, peak_times):
    # Find onset times within any of the time ranges
    valid_onset_times = []
    valid_offset_times = []

    for onset, offset in zip(onset_times, offset_times):
        for time_point in peak_times:
            if onset <= time_point <= offset:
                valid_onset_times.append(onset)
                valid_offset_times.append(offset)
                break  # Move to the next event range as this one is valid.
    return np.array(valid_onset_times), np.array(valid_offset_times)


def create_motor_binaries(stimulus_list, event_activity, onset_idx_ca_rec, offset_idx_ca_rec, ca_time_axis):
    stimulus_names = event_activity['stimulus'].unique()
    binaries = pd.DataFrame()
    for s_name in stimulus_names:
        idx = event_activity['stimulus'] == s_name
        event_binary = create_binary_from_events(
            onset_idx_ca_rec[idx], offset_idx_ca_rec[idx], ca_time_axis
        )
        binaries[s_name] = event_binary

    # Check if there are any stimulus types without motor events
    # Create an all zero regressor
    for s_n in stimulus_list:
        if s_n not in stimulus_names:
            binaries[s_n] = np.zeros_like(ca_time_axis)

    return binaries


def ventral_root_detection(vr_trace, vr_time, ca_time_axis, sw, stimulus_protocol, settings):
    # Ignore first secs of recording (= set them to zero)
    vr_trace = ignore_start_and_end(vr_trace, vr_time, settings['ignore_first_seconds'], settings['ignore_last_seconds'])

    # Replace high values from the VR Trace with mean values
    vr_trace = replace_high_values(vr_trace, th_remove=settings['remove_high_values'])

    # High Pass Filter VR
    vr_trace = filter_high_pass(
        vr_trace, cutoff=settings['vr_trace_low_pass_cutoff'], fs=settings['vr_sampling_rate'], order=4
    )

    # Compute ventral root envelope
    env = envelope(vr_trace, settings['vr_sampling_rate'], freq=settings['envelope_low_pass_cut_off'])

    # Smooth envelope with moving average filter
    env_fil = moving_average_filter(env, window=settings['env_moving_average_window'])

    # Compute Z-Scores
    env_z = z_transform(env_fil)

    # Detect Peaks in envelope, for later validation
    env_z_peaks = detect_peaks(
        env_z,
        height=settings['detection_threshold'],
        distance=int(settings['vr_sampling_rate'] * 0.5),
        width=None,
        prominence=settings['peak_prominence'],
        threshold=None,
    )
    env_z_peaks_times = env_z_peaks['peaks'] / settings['vr_sampling_rate']

    # Create Binary by thresholding the envelope signal
    binary = np.zeros_like(env_z)
    binary[env_z > settings['detection_threshold']] = 1

    # Find onsets and offsets of ventral root activity
    onset_times, offset_times, onset_idx, offset_idx = find_onsets_offsets(binary, vr_time)

    # Check Symmetry
    if settings['check_symmetry'] is not None:
        onset_times, offset_times = validate_signal_symmetry(
            vr_trace, onset_idx, offset_idx, onset_times, offset_times,
            parameter=settings['check_symmetry_parameter'],
            method=settings['check_symmetry'],
        )

    # Check if peaks and binary thresholding match
    onset_times, offset_times = check_peaks(onset_times, offset_times, env_z_peaks_times)

    # Check if the detected times do not fall outside the Ca recording time
    onset_times, offset_times = check_if_data_inside_boundaries(onset_times, offset_times, ca_time_axis)

    # Check if number of onset times is the same as number of offset times
    if len(offset_times) != len(onset_times):
        print('')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('WARNING: Number of onset times and offset times do not match!')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('')
        embed()
        exit()

    # Remove events that are too long
    # check for motor activity that is too long (artifacts due to concatenating recordings) and remove it
    onset_times, offset_times = remove_too_long_events(onset_times, offset_times, duration_th=settings['duration_threshold'])

    # Remove events that are too short
    onset_times, offset_times = remove_too_short_events(onset_times, offset_times, duration_th=settings['min_duration_threshold'])

    # Check that there are no offset times smaller than corresponding onset times
    if ((offset_times-onset_times) <= 0).any():
        print('')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('WARNING: FOUND OFFSET TIME TO BE SMALLER THAN ONSET TIME !!!!')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('')
        embed()
        exit()

    # Merge events that are too close to each other (they are considered to be just one event)
    if settings['min_time_merging'] > 0:
        onset_times, offset_times = merge_events(
            onset_times, offset_times, threshold=settings['min_time_merging']
        )

    # Find corresponding time point in the ca time axis (always go into the past)
    _, onset_idx_ca_rec = find_closest_past_times(onset_times, ca_time_axis)
    _, offset_idx_ca_rec = find_closest_past_times(offset_times, ca_time_axis)

    # Compute Signal Power of each event
    vr_events_power, vr_events_energy = event_power(
        vr_trace, settings['vr_sampling_rate'], onset_times, offset_times, window_length=51, polyorder=3
    )
    vr_env_power, vr_env_integral = envelope_roc(env_fil, settings['vr_sampling_rate'], onset_times, offset_times)

    # Put all into a data frame
    event_activity = pd.DataFrame()
    event_activity['start_idx'] = onset_idx_ca_rec
    event_activity['end_idx'] = offset_idx_ca_rec
    event_activity['start_time'] = onset_times
    event_activity['end_time'] = offset_times
    event_activity['duration'] = offset_times - onset_times
    event_activity['power'] = vr_events_power
    event_activity['energy'] = vr_events_energy
    event_activity['env_roc'] = vr_env_integral
    event_activity['env_power'] = vr_env_power

    # Create a Binary Trace from events in Ca Imaging Resolution
    event_binary = create_binary_from_events(onset_idx_ca_rec, offset_idx_ca_rec, ca_time_axis)

    # Put Event Binary into data frame
    vr_binary_df = pd.DataFrame()
    vr_binary_df[sw] = event_binary

    # Check Overlap with Stimulus
    if stimulus_protocol is not None:
        stimulus_types = [
            'spontaneous',
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
        vr_overlaps = check_stimulus_overlap(stimulus_protocol, onset_times, settings=settings['stimulus_overlap'])

        # vr_overlaps = check_stimulus_overlap(stimulus_protocol, onset_times,
        #                                      stimulus_earlier=settings['before_stimulus'],
        #                                      stimulus_later=settings['after_stimulus'])
        event_activity['stimulus'] = vr_overlaps[:, 1]

        # Create a Binary Traces in Ca Imaging Resolution
        event_binary_all = create_motor_binaries(stimulus_types, event_activity, onset_idx_ca_rec, offset_idx_ca_rec, ca_time_axis)

        # Create a Binary Trace from all stimulus associated events in Ca Imaging Resolution
        event_binary_stimulus = event_binary_all.drop(columns=['spontaneous']).sum(axis=1)

        # Collect Data
        vr_binary_spontaneous_df = pd.DataFrame()
        vr_binary_spontaneous_df[sw] = event_binary_all['spontaneous']
        vr_binary_stimulus_df = pd.DataFrame()
        vr_binary_stimulus_df[sw] = event_binary_stimulus
        return event_activity, vr_binary_df, env_z, vr_binary_spontaneous_df, vr_binary_stimulus_df, event_binary_all

    return event_activity, vr_binary_df, env_z, None, None, None


def validate_detection(vr_recordings, vr_time, ca_time_axis, settings,  stimulus_protocols_dir, stimulus_trace_dir, sw):
    vr = vr_recordings[sw]

    if stimulus_protocols_dir is not None:
        sw_stimulus_protocol = pd.read_csv(f'{stimulus_protocols_dir}/{sw}_stimulus_protocol.csv', header=None)
        sw_stimulus_trace = pd.read_csv(stimulus_trace_dir)[sw]  # for plotting
    else:
        sw_stimulus_protocol = None
        sw_stimulus_trace = None

    print('')
    print(f'Sweep: {sw}')
    print('')
    event_activity, vr_binary_df, vr_env, vr_binary_spontaneous_df, vr_binary_stimulus_df, event_binary_all = ventral_root_detection(
        vr, vr_time, ca_time_axis, sw, sw_stimulus_protocol, settings=settings
    )
    print('')
    print(event_activity)
    print('')
    print(f'VR Sweep {sw} done')
    print('')
    vr_fil = filter_high_pass(vr, cutoff=settings['vr_trace_low_pass_cutoff'], order=4, fs=settings['vr_sampling_rate'])
    test_plot(vr_fil, ca_time_axis, vr_time, event_activity, vr_binary_df, vr_env, sw_stimulus_trace,
              settings['detection_threshold'])


def main():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_dir = f'{base_dir}/ventral_root/selected_sweeps_aligned_vr_recordings.h5'
    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'
    stimulus_protocols_dir = f'{base_dir}/protocols'
    stimulus_trace_dir = f'{base_dir}/stimulus_traces/stimulus_traces.csv'

    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()
    ca_time_axis = np.linspace(0, ca_df_f.shape[0] / ca_sampling_rate, ca_df_f.shape[0])

    # Settings
    detection_validation = False
    check_stimulus = True
    settings = {
        'vr_sampling_rate': 10000,  # VR Sampling Rate in Hz
        'vr_trace_low_pass_cutoff': 500,  # Low Pass Filter Cut Off for Vr Trace (remove slow oscillations)
        'detection_threshold': 1.5,  # in SD
        'duration_threshold': 2,  # in seconds. Events longer than that will be ignored
        'min_duration_threshold': 0.1,  # in seconds. Events shorter than that will be ignored
        'envelope_low_pass_cut_off': 5,  # Low Pass Filter Cut Off used in envelope computation in Hz (5)
        'ca_sampling_rate': ca_sampling_rate,
        'ignore_first_seconds': 5,  # ignore the first seconds of the vr recording
        'ignore_last_seconds': 2,  # ignore the last seconds of the vr recording
        'remove_high_values': 20,  # in SD
        'env_moving_average_window': 20,  # in samples: The Window for the moving average filter for the envelope signal
        'min_time_merging': 0,  # minimum time in seconds. If two events are too close merge them. Ignore if set to 0
        'peak_prominence': 1,  # scipy find_peaks prominence. Is used to identify real signals and noise.
        'check_symmetry': 'energy',  # Check Symmetry Method (None, stats, freq, energy
        'check_symmetry_parameter': 0.4,  # Depends on Method

        'stimulus_overlap': {  # buffer before and after stimulus period for classifying swim events in seconds
            'moving_target_01': [1, 1],
            'moving_target_02': [1, 1],
            'grating_appears': [1, 3],
            'grating_disappears': [1, 3],
            'grating_0': [1, 1],
            'grating_180': [1, 1],
            'bright_loom': [1, 3],
            'dark_loom': [1, 3],
            'bright_flash': [1, 3],
            'dark_flash': [1, 3]
        }
    }

    # Load selection (good sweeps) of aligned VR data
    vr_recordings = pd.read_hdf(vr_dir, key='data')
    vr_time = np.linspace(0, vr_recordings.shape[0] / settings['vr_sampling_rate'], vr_recordings.shape[0])

    print('++++ INFO ++++')
    print('SETTINGS')
    for key in settings:
        print(f'{key}: {settings[key]}')
    print('++++++++++++++')
    print('')
    print('Please wait ...')
    print('')

    sweeps = list(vr_recordings.keys())
    print(f'---- Number of sweeps: {len(sweeps)} ----')
    # Check Detection Validation

    if detection_validation:
        sweep_nr = -2
        print(f'Sweep: {sweeps[sweep_nr]}')
        print(f'{sweep_nr} / {len(sweeps)}')
        validate_detection(vr_recordings.copy(), vr_time, ca_time_axis, settings, stimulus_protocols_dir, stimulus_trace_dir, sweeps[sweep_nr])
        exit()

    # Start Ventral Root Activity (Event) Detection
    # vr_binaries = dict()
    vr_binaries = pd.DataFrame()
    vr_binaries_spontaneous = pd.DataFrame()
    vr_binaries_stimulus = pd.DataFrame()
    vr_events = dict()
    vr_binaries_all = dict()

    # vr_merged_binaries = pd.DataFrame()
    # vr_merged_binaries_spontaneous = pd.DataFrame()
    # vr_merged_binaries_stimulus = pd.DataFrame()
    # vr_merged_events = dict()

    for sw in vr_recordings:
        print(f'START PROCESSING: VR Sweep {sw} ...')
        vr = vr_recordings[sw].to_numpy()

        if check_stimulus:
            sw_stimulus_protocol = pd.read_csv(f'{stimulus_protocols_dir}/{sw}_stimulus_protocol.csv', header=None)
        else:
            sw_stimulus_protocol = None

        event_activity, vr_binary_df, _, vr_binary_spontaneous_df, vr_binary_stimulus_df, vr_binary_all = ventral_root_detection(
            vr, vr_time, ca_time_axis, sw, sw_stimulus_protocol, settings=settings
        )

        vr_events[sw] = event_activity
        vr_binaries[sw] = vr_binary_df

        if check_stimulus:
            vr_binaries_spontaneous[sw] = vr_binary_spontaneous_df
            vr_binaries_stimulus[sw] = vr_binary_stimulus_df
            vr_binaries_all[sw] = vr_binary_all

        # settings['min_time_merging'] = 2  # secs
        # m_event_activity, m_vr_binary_df, _, m_vr_binary_spontaneous_df, m_vr_binary_stimulus_df = ventral_root_detection(
        #     vr, vr_time, ca_time_axis, sw, sw_stimulus_protocol, settings=settings
        # )
        # vr_merged_events[sw] = m_event_activity
        # vr_merged_binaries[sw] = m_vr_binary_df
        # if check_stimulus:
        #     vr_merged_binaries_spontaneous[sw] = m_vr_binary_spontaneous_df
        #     vr_merged_binaries_stimulus[sw] = m_vr_binary_stimulus_df

    # Store Results to HDD
    save_dict_as_hdf5(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_events.h5', vr_events)
    save_dict_as_hdf5(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_all_binaries.h5', vr_binaries_all)
    vr_binaries.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries.csv', index=False)
    vr_binaries_spontaneous.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries_spontaneous.csv', index=False)
    vr_binaries_stimulus.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries_stimulus.csv', index=False)

    # save_dict_as_hdf5(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_events_merged.h5', vr_merged_events)
    # vr_merged_binaries.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries_merged.csv', index=False)
    # vr_merged_binaries_spontaneous.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries_spontaneous_merged.csv', index=False)
    # vr_merged_binaries_stimulus.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries_stimulus_merged.csv', index=False)

    print('VR DETECTION FINISHED!')


if __name__ == '__main__':
    import timeit
    n = 1
    result = timeit.timeit(stmt='main()', globals=globals(), number=n)
    # calculate the execution time
    # get the average execution time
    print(f"Execution time is {(result/60) / n: .2f} minutes")
