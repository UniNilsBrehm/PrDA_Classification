import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import tifffile as tiff
from scipy import signal
from scipy.stats import gaussian_kde
from IPython import embed


def binary_to_gaussian(binary, sampling_rate, sigma=1.0):
    """
    Convert a hard-edge binary trace to a smooth Gaussian bell-shaped trace.

    Parameters:
        binary (np.ndarray): Binary trace (1D array containing 0s and 1s).
        sigma (float): Standard deviation of the Gaussian kernel (in seconds).
        sampling_rate (float): Sampling rate of the binary trace (samples per second).

    Returns:
        np.ndarray: Gaussian-smoothed trace.
    """
    # Ensure binary is a numpy array
    binary = np.asarray(binary)

    # Validate that the binary trace contains only 0s and 1s
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("The binary trace must contain only 0s and 1s.")

    # Create a time vector for the Gaussian kernel
    kernel_size = int(6 * sigma * sampling_rate)  # Kernel spans ~6 sigma (3 on each side)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd for symmetry
    t = np.linspace(-3 * sigma, 3 * sigma, kernel_size)

    # Create the Gaussian kernel
    gaussian_kernel = np.exp(-0.5 * (t / sigma) ** 2)
    gaussian_kernel /= np.sum(gaussian_kernel)  # Normalize to make it a proper Gaussian

    # Convolve the binary trace with the Gaussian kernel
    gaussian_trace = np.convolve(binary, gaussian_kernel, mode='same')  # Same size as input

    return gaussian_trace


def broaden_binary_trace(binary, width=1):
    """
    Broaden the regions where the binary trace equals 1.

    Parameters:
        binary (np.ndarray): Binary trace (1D array containing 0s and 1s).
        width (int): Number of steps to extend the 1s on either side.

    Returns:
        np.ndarray: Broadened binary trace.
    """
    # Ensure binary is a numpy array
    binary = np.asarray(binary)

    # Validate that the binary trace contains only 0s and 1s
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("The binary trace must contain only 0s and 1s.")

    # Create an extended binary array
    broadened_binary = np.copy(binary)

    for i in range(1, width + 1):
        # Shift the binary trace to the left and right
        broadened_binary[:-i] = np.maximum(broadened_binary[:-i], binary[i:])
        broadened_binary[i:] = np.maximum(broadened_binary[i:], binary[:-i])

    return broadened_binary


def broaden_binary_trace_asymmetrical(binary, left_width=1, right_width=1):
    """
    Broaden the regions where the binary trace equals 1,
    extending a different number of steps to the left and right.

    Parameters:
        binary (np.ndarray): 1D array of 0s and 1s.
        left_width (int): Number of steps to extend the 1s to the left.
        right_width (int): Number of steps to extend the 1s to the right.

    Returns:
        np.ndarray: The broadened binary trace.
    """
    binary = np.asarray(binary)
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("The binary trace must contain only 0s and 1s.")

    # Start with a copy of the original binary trace.
    broadened_binary = np.copy(binary)

    # Extend to the left: if a 1 occurs at a later index, copy it to earlier indices.
    for i in range(1, left_width + 1):
        broadened_binary[:-i] = np.maximum(broadened_binary[:-i], binary[i:])

    # Extend to the right: if a 1 occurs at an earlier index, copy it to later indices.
    for i in range(1, right_width + 1):
        broadened_binary[i:] = np.maximum(broadened_binary[i:], binary[:-i])

    return broadened_binary


def get_rois_of_one_sweep(sw, rois_per_sweep):
    sw_rois = rois_per_sweep[sw].astype('str')
    return sw_rois


def get_rois_per_sweep(ca_labels):
    sweeps = ca_labels.iloc[1, :].unique()
    sw_rois = dict()
    for sw in sweeps:
        idx = ca_labels.iloc[1, :].isin([sw])
        sw_rois[sw] = idx.index[idx].to_numpy().astype('int')
    return sw_rois


def create_regressors_from_binary(binary, cif, delta=False, norm=False) -> object:
    """
    Create a regressor trace from a binary trace and a calcium impulse function (CIF).

    Parameters:
        binary (np.ndarray): Binary trace (1D array) where each entry is 0 or 1.
        cif (np.ndarray): Calcium impulse response function (1D array).
        delta (bool): If True, converts the binary trace into a delta function (only stimulus onset).
        norm (bool): If True, normalizes the final regressor trace using min-max normalization.

    Returns:
        np.ndarray: Regressor trace with the same size as the input binary trace.
    """
    # Ensure binary is a numpy array
    binary = np.asarray(binary)

    # Validate that the binary trace is 1D
    if binary.ndim != 1:
        raise ValueError("The binary input must be a 1D array.")

    # Validate that the CIF is a 1D array
    cif = np.asarray(cif)
    if cif.ndim != 1:
        raise ValueError("The calcium impulse response function (CIF) must be a 1D array.")

    if delta:
        # Convert binary trace to a delta function (only detect stimulus onsets)
        binary = np.diff(binary, prepend=0)  # Find changes in the binary signal
        binary[binary < 0] = 0  # Remove negative changes (offsets)

    # Convolve the binary trace with the calcium impulse response (CIF)
    reg = np.convolve(binary, cif, mode='full')  # 'full' ensures the entire convolution is computed

    # Truncate the result to match the size of the binary trace
    reg_final = reg[:binary.shape[0]]

    if norm:
        # Apply min-max normalization to the final regressor trace
        reg_final = norm_min_max(reg_final)

    return reg_final


def norm_min_max(arr):
    """
    Normalize an array to the range [0, 1].

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Min-max normalized array.
    """
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val == 0:
        return arr  # Avoid division by zero; return original array
    return (arr - min_val) / (max_val - min_val)


def calcium_impulse_response(tau_rise=1, tau_decay=3, amplitude=1.0, sampling_rate=1000, threshold=1e-5, norm=False):
    """
    Generates a Calcium Impulse Response Function, dynamically truncating when the signal fades below a threshold.

    Parameters:
        tau_rise (float): Time constant for the rise phase (in seconds).
        tau_decay (float): Time constant for the decay phase (in seconds).
        amplitude (float): Scaling factor for the response function.
        sampling_rate (float): Sampling rate in Hz (samples per second).
        threshold (float): Threshold below which the response is considered negligible.
        norm (bool): Normalize to max = 1.

    Returns:
        np.ndarray: Time vector and Calcium Impulse Response.
    """
    # Calculate the effective duration of the response
    duration = -tau_decay * np.log(threshold / amplitude)

    # Create a time vector up to the calculated duration
    dt = 1.0 / sampling_rate  # Time step
    t = np.arange(0, duration, dt)  # Time vector

    # Compute the double exponential function
    response = amplitude * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))

    # Ensure the response is zero for negative time values
    response[t < 0] = 0

    if norm:
        response /= np.max(response)

    return t, response


def align_traces_via_padding_df(dataframe, delays, sampling_rate):
    """
    Aligns all traces (columns) in a pandas DataFrame to a reference signal by applying padding based on the given delays.

    Parameters:
        dataframe (pd.DataFrame): DataFrame where each column is a signal trace to be aligned.
        delays (dict): A dictionary where keys are column names and values are delays (in seconds).
                       Positive delay indicates the signal started later than the reference.
                       Negative delay indicates the signal started earlier than the reference.
        sampling_rate (float): The sampling rate of the signals (in samples per second).

    Returns:
        pd.DataFrame: A new DataFrame with the aligned signal traces.
    """

    aligned_traces = {}

    for column in dataframe.columns:
        trace = dataframe[column].values
        delay = delays.get(column, 0).item()  # Default to 0 if no delay is specified for this column
        num_samples_to_shift = int(abs(delay) * sampling_rate)
        if num_samples_to_shift > 0:
            if delay > 0:
                # Positive delay: Pad zeros at the start
                aligned_trace = np.concatenate((
                    np.zeros(num_samples_to_shift),  # Add zeros to the beginning
                    trace[:-num_samples_to_shift]  # Truncate the end
                ))
            elif delay < 0:
                # Negative delay: Pad zeros at the end
                aligned_trace = np.concatenate((
                    trace[num_samples_to_shift:],  # Truncate the beginning
                    np.zeros(num_samples_to_shift)  # Add zeros to the end
                ))
            else:
                # No alignment needed if delay is zero or smaller than 1/sampling_rate
                aligned_trace = trace
        else:
            # No alignment needed if delay is zero or smaller than 1/sampling_rate
            aligned_trace = trace

        # Add the aligned trace to the result dictionary
        aligned_traces[column] = aligned_trace

    # Create a new DataFrame with the aligned traces
    aligned_df = pd.DataFrame(aligned_traces)

    return aligned_df


def align_traces_via_padding(trace, delay, sampling_rate):
    """
    Aligns a signal trace to a reference signal by applying padding based on the given delay.

    Parameters:
        trace (array-like): The signal trace to be aligned.
        delay (float): The time lag (in seconds) between the signal and the reference.
                      Positive delay indicates the signal started later than the reference.
                      Negative delay indicates the signal started earlier than the reference.
        sampling_rate (float): The sampling rate of the signals (in samples per second).

    Returns:
        array-like: The aligned signal trace with padding applied as necessary.
    """

    # Calculate the number of samples to shift, based on the delay and sampling rate
    num_samples_to_shift = int(abs(delay) * sampling_rate)

    # Check if delay is larger than the sampling resolution
    # Otherwise it is meaningless to align the signals
    if num_samples_to_shift > 0:
        # Adjust the signal based on the delay
        if delay > 0:
            # If delay is positive, the signal started later than the reference.
            # Pad zeros at the start of the signal to shift it to the right on the time axis.
            padded_signal = np.concatenate((
                np.zeros(num_samples_to_shift),  # Add zeros to the beginning
                trace[:-num_samples_to_shift]  # Truncate the signal at the end
            ))

        elif delay < 0:
            # If delay is negative, the signal started earlier than the reference.
            # Pad zeros at the end of the signal to shift it to the left on the time axis.
            padded_signal = np.concatenate((
                trace[num_samples_to_shift:],  # Truncate the signal at the beginning
                np.zeros(num_samples_to_shift)  # Add zeros to the end
            ))
        else:
            # If there is no delay, the signal is already aligned with the reference.
            padded_signal = trace
    else:
        # If delay is smaller than 1/sampling_rate, the signal cannot be aligned more with the reference.
        padded_signal = trace

    # Return the aligned (padded) signal
    return padded_signal


def filter_low_pass(data, cutoff, fs, order=2):
    sos = butter_filter_design('lowpass', cutoff, fs, order=order)
    return signal.sosfiltfilt(sos, data, axis=0)


def filter_high_pass(data, cutoff, fs, order=2):
    sos = butter_filter_design('highpass', cutoff, fs, order=order)
    return signal.sosfiltfilt(sos, data, axis=0)


def butter_filter_design(filter_type, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.9999
    sos = signal.butter(order, normal_cutoff, btype=filter_type, output='sos')
    return sos


def load_stimulus_protocols(protocol_dir):
    file_list = os.listdir(protocol_dir)
    protocols = list()
    for f in file_list:
        f_dir = f'{protocol_dir}/{f}'
        protocols.append(pd.read_csv(f_dir, header=None))
    return protocols


def load_hdf5_as_data_frame(hdf5_file):
    return pd.read_hdf(hdf5_file, key='data')


def save_dat_frame_as_hdf5(hdf5_file, df):
    df.to_hdf(hdf5_file, key='data', mode='w')


def load_hdf5_as_dict(hdf5_file):
    # Read DataFrames from the HDF5 file
    retrieved_df_dict = {}
    with pd.HDFStore(hdf5_file, mode='r') as store:
        for key in store.keys():
            retrieved_df_dict[key.strip('/')] = store[key]
    return retrieved_df_dict


def save_dict_as_hdf5(hdf5_file, df_dict):
    # Save DataFrames to the HDF5 file
    with pd.HDFStore(hdf5_file, mode='w') as store:
        for key, df in df_dict.items():
            store.put(key, df)


def z_transform(data):
    result = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return result


def moving_average_filter(data, window):
    return np.convolve(data, np.ones(window) / window, mode='same')


def convert_time_stamps_to_secs(data, method=0):
    if method == 0:
        # INPUT: hhmmss.ms (163417.4532)
        s = str(data)
        in_secs = int(s[:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])
    elif method == 1:
        # INPUT: hh:mm:ss
        s = data
        in_secs = int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:])
    else:
        in_secs = None
    return in_secs


def delta_f_over_f(data, fr, fbs_per=5, window=None):
    """ Compute dF/F
    data: pandas data frame (each column is one ROI)
    fr: sampling rate
    fbs_per: percentile (in percent: e.g. 5 %) for calculating the baseline
    window: the sliding window in secs for calculating the dynamic baseline
    """
    if window is None:
        fbs = np.percentile(data, fbs_per, axis=0)
        fbs_df = pd.DataFrame(fbs, index=data.keys(), columns=['fluo baseline'])

    else:
        per_window = int(window * fr)
        quant = fbs_per / 100
        fbs = data.rolling(window=per_window, center=True, min_periods=0).quantile(quant)
        fbs_df = fbs
    df = (data - fbs) / fbs
    return df, fbs_df


def load_ca_data_headers_only(file_dir):
    # Imaging Data
    # Each Col is one ROI
    # Row 0: Fish Number
    # Row 1: Sweep Number
    # Row 2: ROI Number
    # Row 3: Y-X Position
    # Row 4: Z Position (Plane Number)
    # Row 5-N: Fluorescence Values
    validate_directory(file_dir)
    return pd.read_csv(file_dir, header=None, index_col=False, nrows=5)


def load_raw_ca_data_only(file_dir):
    validate_directory(file_dir)
    return pd.read_csv(file_dir, header=None, index_col=False, skiprows=5)


def load_ca_meta_data(file_dir):
    validate_directory(file_dir)
    return pd.read_csv(file_dir, index_col=0)


def validate_directory(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")


def simple_interpolation(data, new_size, test_plot=False):
    # Original time arrays
    time_fewer = np.linspace(0, 1, len(data))
    time_more = np.linspace(0, 1, new_size)

    # Interpolating data_fewer_samples to match the length of data_more_samples
    interpolated_data = np.interp(time_more, time_fewer, data)

    # plot
    if test_plot:
        plt.plot(data, 'k')
        plt.plot(interpolated_data, 'r')
        plt.show()

    return interpolated_data


def pickle_stuff(file_name, data=None):
    # check if the dir is a PATH
    if str(type(file_name)) == "<class 'pathlib.WindowsPath'>":
        file_name = file_name.as_posix()

    if not file_name.endswith('.pickle'):
        print('ERROR: File name must end with ".pickle"')
        return None

    if data is None:
        with open(file_name, 'rb') as handle:
            result = pickle.load(handle)
        return result
    else:
        with open(file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True


def smooth_histogram(data, bandwidth=0.3, res=1000):
    # Compute KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Create x axis
    x_vals = np.linspace(min(data), max(data), res)

    # Compute smoothed values
    kde_vals = kde(x_vals)

    return x_vals, kde_vals


def detect_peaks(data, height=None, distance=None, width=None, prominence=None, threshold=None):
    """
    Detect peaks in a signal with adjustable criteria.

    Parameters:
        data (array-like): The input signal (1D array).
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
        data, height=height, distance=distance, width=width, prominence=prominence, threshold=threshold
    )

    # Extract width properties if peaks were found
    if len(peaks) > 0:
        widths_result = peak_widths(data, peaks, rel_height=0.5)
        properties['widths'] = widths_result[0]
        properties['width_heights'] = widths_result[1]
        properties['left_ips'] = widths_result[2]
        properties['right_ips'] = widths_result[3]

    return {'peaks': peaks, 'properties': properties}


def load_tiff_recording(file_name, flatten=False):
    all_frames = []

    with tiff.TiffFile(file_name) as tif:
        for i, series in enumerate(tif.series):
            # print(f"Series {i} shape: {series.shape}")
            data = series.asarray()
            all_frames.append(data)

    # Flatten the list if needed
    if flatten:
        frames = np.concatenate(all_frames, axis=0).view(np.uint16)
    else:
        frames = np.array(all_frames)

    return frames


def generate_sweep_directory_list(prefix, sweep_range=(1, 20), numbering='02'):
    folder_names = [f"{prefix}_{i:{numbering}}" for i in range(sweep_range[0], sweep_range[1])]
    return folder_names


def generate_folders(folder_names, parent_dir):
    for folder in folder_names:
        folder_path = os.path.join(parent_dir, folder)
        os.makedirs(folder_path, exist_ok=True)  # won't error if folder already exists
