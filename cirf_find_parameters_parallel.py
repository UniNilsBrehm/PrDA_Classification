import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import calcium_impulse_response, create_regressors_from_binary, detect_peaks, save_dict_as_hdf5
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import Config
from IPython import embed


def generate_simulated_calcium_data(
        num_neurons=10,
        duration_minutes=15,
        sampling_rate=2.7,
        event_rate_per_min=2,
        tau_rise=0.3,
        tau_decay=1.5,
        noise_std=0.05,
        baseline_drift_amp=0.2,
        seed=None
):
    """
    Generate a Pandas DataFrame of synthetic calcium traces.

    Parameters:
        num_neurons (int): Number of neurons (columns in DataFrame)
        duration_minutes (float): Total trace length in minutes
        sampling_rate (float): Sampling rate in Hz
        event_rate_per_min (float): Avg number of events per neuron per minute
        tau_rise (float): Tau rise for double exponential (sec)
        tau_decay (float): Tau decay for double exponential (sec)
        noise_std (float): Std dev of Gaussian noise
        baseline_drift_amp (float): Amplitude of baseline drift (sinusoidal)
        seed (int or None): Random seed for reproducibility

    Returns:
        DataFrame: Rows = timepoints, Columns = neurons
    """
    if seed is not None:
        np.random.seed(seed)

    duration_sec = duration_minutes * 60
    num_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, num_samples)

    def generate_event(t, A, tau_rise, tau_decay, t0):
        response = np.zeros_like(t)
        valid_idx = t >= t0
        t_shifted = t[valid_idx] - t0
        response[valid_idx] = A * (np.exp(-t_shifted / tau_decay) - np.exp(-t_shifted / tau_rise))
        return response

    all_traces = []
    for neuron_idx in range(num_neurons):
        trace = np.zeros_like(t)
        num_events = np.random.poisson(event_rate_per_min * duration_minutes)

        event_times = np.sort(np.random.uniform(5, duration_sec - 5, size=num_events))
        for t0 in event_times:
            amplitude = np.random.uniform(0.5, 1.5)  # Variable amplitude per event
            event = generate_event(t, amplitude, tau_rise, tau_decay, t0)
            trace += event

        # Add baseline drift (slow sinusoid)
        baseline_drift = baseline_drift_amp * np.sin(2 * np.pi * t / (duration_sec / 2))
        # Add noise
        noise = np.random.normal(0, noise_std, size=t.shape)

        final_trace = trace + baseline_drift + noise
        all_traces.append(final_trace)

    df = pd.DataFrame(np.array(all_traces).T, columns=[f'neuron_{i + 1}' for i in range(num_neurons)])
    # df.index = pd.Index(t, name='time_sec')
    return df


def find_ca_signals_cross_correlation(ca_trace, ca_sampling_rate):
    # Generate Calcium Impulse Response Function
    _, cif = calcium_impulse_response(tau_rise=3, tau_decay=7, amplitude=1.0,
                                      sampling_rate=ca_sampling_rate, threshold=1e-5, norm=True)
    # Number of samples in the original signals
    n = len(ca_trace)  # Assumes both signals are the same length

    binary = np.zeros_like(ca_trace)
    center = int(n/2)
    binary[center] = 1
    reg = create_regressors_from_binary(binary, cif, delta=True, norm=True)

    # Cross Correlation
    corr = np.correlate(ca_trace, reg, mode='full')

    # corr_th = np.mean(corr) + np.std(corr)
    corr_th = np.percentile(corr, 75)
    peaks_info = detect_peaks(corr, height=corr_th, distance=20, width=1, prominence=1.5, threshold=None)

    corr_peaks = peaks_info['peaks']
    ca_peaks = (corr_peaks - (n/2)).astype('int')

    return ca_peaks


def find_ca_signals(ca_trace, ca_sampling_rate):
    from utils import filter_low_pass
    ca_trace_fil = filter_low_pass(ca_trace, cutoff=1.0, fs=ca_sampling_rate, order=2)
    # th = np.percentile(ca_trace_fil, 75)
    th = np.mean(ca_trace_fil) + np.std(ca_trace_fil)

    peaks_info = detect_peaks(ca_trace_fil, height=th, distance=10, width=2, prominence=0.2, threshold=None)
    ca_peaks = peaks_info['peaks']

    # plt.plot(ca_trace_fil, 'k')
    # plt.plot(ca_peaks, ca_trace_fil[ca_peaks], 'rx')
    # plt.plot([0, ca_trace_fil.shape[0]], [th, th], 'r--')
    # plt.show()
    # embed()
    # exit()

    return ca_peaks


def compute_r2(data, fitted):
    # Calculate R²
    ss_res = np.sum((data - fitted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def shifted_double_exponential(t, A, tau_rise, tau_decay, t0):
    response = np.zeros_like(t)
    valid_idx = t >= t0
    t_shifted = t[valid_idx] - t0
    response[valid_idx] = A * (np.exp(-t_shifted / tau_decay) - np.exp(-t_shifted / tau_rise))
    return response


def fit_double_exp_peak(ca_cutout, t, show_plot=False):
    try:
        bounds = ([0, 0, 0, 0], [np.inf, 100, 100, np.max(t)])
        p0 = [1.0, 2, 3, np.max(t)/2]
        # popt, _ = curve_fit(shifted_double_exponential, t, ca_cutout, p0=p0, bounds=bounds, maxfev=5000)
        popt, _ = curve_fit(shifted_double_exponential, t, ca_cutout, bounds=bounds, maxfev=2000)
        # popt, _ = curve_fit(shifted_double_exponential, t, ca_cutout, bounds=bounds, maxfev=2000)
        fitted = shifted_double_exponential(t, *popt)
        r2 = compute_r2(ca_cutout, fitted)

        # show_plot = True
        if show_plot:
            plt.figure(figsize=(8, 4))
            plt.plot(t, ca_cutout, 'k', lw=3, label='Data')
            plt.plot(t, fitted, 'r--', lw=2, label='Double Exp Fit')
            plt.xlabel('Time (s)')
            plt.ylabel('Response')
            plt.legend()
            plt.title(f'Fit R² = {r2:.3f}')
            plt.grid(True)
            plt.show()
            exit()

        return np.append(popt, r2), ca_cutout
    except RuntimeError:
        print('Fit failed for one peak.')
        return None, ca_cutout


def plot_compare_detections(peaks_01, peaks_02, data_trace):
    fig, ax = plt.subplots()
    ax.plot(data_trace, 'k')
    ax.plot(peaks_01, data_trace[peaks_01], 'rx', label='Detection 1')
    ax.plot(peaks_02, data_trace[peaks_02], 'bx', label='Detection 2')
    plt.legend()
    plt.show()


def find_optimal_cirf_parallel(ca_data, ca_sampling_rate, show_plot=False):
    double_exp_parameters = []
    # ca_traces_collection = []

    before = 0  # if onset is detected (e.g. with cross correlation)
    after = 40
    # before = 20  # if max peak is detected
    # after = 50

    fit_jobs = []

    print("Preparing fit jobs...")
    # Prepare jobs
    for k, ca_trace in enumerate(ca_data.T, start=1):
        print(f'Queueing ROI {k} / {ca_data.shape[1]}', end='\r')
        ca_peaks = find_ca_signals_cross_correlation(ca_trace, ca_sampling_rate)
        # ca_peaks2 = find_ca_signals(ca_trace, ca_sampling_rate)
        for peak in ca_peaks:
            try:
                ca_cutout = ca_trace[peak-before:peak+after]
                if len(ca_cutout) < after:  # Skip if cutout too short
                    continue
                ca_cutout = ca_cutout - np.min(ca_cutout[:10])
                t = np.linspace(0, len(ca_cutout) / ca_sampling_rate, len(ca_cutout))
                fit_jobs.append((ca_cutout, t, show_plot))
            except:
                continue

    print(f"Total fit jobs: {len(fit_jobs)}")

    # Run in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(fit_double_exp_peak, *job) for job in fit_jobs]

        for idx, future in enumerate(as_completed(futures), start=1):
            result, ca_cutout = future.result()
            if result is not None:
                double_exp_parameters.append(result)
            # ca_traces_collection.append(ca_cutout)
            print(f'Processed {idx} / {len(futures)}', end='\r')

    # Convert to DataFrame
    # ca_traces_collection = pd.DataFrame(ca_traces_collection).transpose()
    double_exp_parameters = pd.DataFrame(double_exp_parameters, columns=['amplitude', 'tau_rise', 'tau_decay', 't0', 'R2'])

    return double_exp_parameters


def run_simulation(ca_sampling_rate):
    n_neurons = 1000
    duration = 15  # in minutes

    tau_values = [
        (0.2, 1.0),  # (tau_rise, tau_decay)
        (1.0, 3.0),
        (2.0, 4.0),
        (3.0, 6.0),
        # (5.0, 10.0),
    ]
    event_rate = 0.5

    # noise_level = {
    #     'low': (0.02, 0.01),  # (noise_std, baseline_drift_amp)
    #     'high': (0.05, 0.05),
    # }
    noise = (0.02, 0.01)

    k = 0
    for taus in tau_values:
        # noise = noise_level['low']
        simulated_data = generate_simulated_calcium_data(
            num_neurons=n_neurons,
            duration_minutes=duration,
            sampling_rate=ca_sampling_rate,
            event_rate_per_min=event_rate,
            tau_rise=taus[0],
            tau_decay=taus[1],
            noise_std=noise[0],
            baseline_drift_amp=noise[1],
        )

        # fig, axs = plt.subplots(5, 1)
        # for k in range(5):
        #     col = np.random.randint(n_neurons)
        #     axs[k].plot(simulated_data.iloc[:, col])
        # plt.show()
        # embed()
        # exit()
        # Test it on simulated Ca Data
        sim_params_df = find_optimal_cirf_parallel(simulated_data.to_numpy(), ca_sampling_rate, show_plot=False)
        sim_settings = pd.DataFrame({
            'tau_rise': taus[0],
            'tau_decay': taus[1],
            'event_rate': event_rate,
            'sampling_rate': ca_sampling_rate,
            'neurons': n_neurons,
            'noise_std': noise[0],
            'noise_baseline_drift_amp': noise[1],
            'duration': duration,
        }, index=[0])

        # Store to HDD
        sim_params_df.to_csv(f'{Config.BASE_DIR}/data/tau_fitting/cirf_fitted_parameters_simulated_run_{k}.csv', index=False)
        sim_settings.to_csv(f'{Config.BASE_DIR}/data/tau_fitting/cirf_fitted_parameters_simulated_run_{k}_settings.csv', index=False)
        # sim_r2_thresh = sim_params_df['R2'].quantile(0.2)  # Keep top 80% fits
        # sim_params_df_final = sim_params_df[sim_params_df['R2'] >= sim_r2_thresh].reset_index(drop=True)
        print('')
        print(f'==== FINISHED RUN: {k} ====')
        k += 1


def plot_simulation():
    import seaborn as sns
    from plotting_utils import hide_axis_spines

    for k in range(4):
        sim_params = pd.read_csv(f'{Config.BASE_DIR}/data/tau_fitting/cirf_fitted_parameters_simulated_run_{k}.csv')
        sim_settings = pd.read_csv(
            f'{Config.BASE_DIR}/data/tau_fitting/cirf_fitted_parameters_simulated_run_{k}_settings.csv')

        # Thresholding by R2
        # sim_r2_thresh = sim_params['R2'].quantile(0.4)  # Keep top 60% fits
        sim_r2_thresh = 0.8
        good_params = sim_params[sim_params['R2'] >= sim_r2_thresh].reset_index(drop=True)
        i = 0
        for param_name in ['tau_rise', 'tau_decay', 'R2']:
            i += 1
            if param_name != 'R2':
                param = good_params[param_name]
            else:
                param= sim_params[param_name]
            fig, ax = plt.subplots()
            sns.histplot(param, kde=True)
            # plt.axvline(param.mean(), color='red', linestyle='--', linewidth=2)
            ax.axvline(param.median(), color='green', linestyle='--', linewidth=2)
            ax.text(0.7, 0.8, f'n={param.shape[0]}', transform=ax.transAxes)
            ax.text(0.7, 0.75, f'median={param.median():.2f} s', transform=ax.transAxes, color='green')
            if param_name != 'R2':
                ax.axvline(sim_settings[param_name].item(), color='red', linestyle='--', linewidth=2)
                ax.text(0.7, 0.7, f'true value={sim_settings[param_name].item()} s', transform=ax.transAxes, color='red')
                ax.set_xlim(0, np.percentile(param, 99)+1)
            else:
                ax.set_xlim(0, 1)
            ax.set_title(f'Distribution of {param_name}')
            ax.set_xlabel(f'{param_name} [s]')
            ax.set_ylabel('Count')
            hide_axis_spines(ax, left=True, right=False, top=False, bottom=True)
            plt.savefig(f'{Config.BASE_DIR}/figures/tau_fitting/simulation_run_{k}_{i}_{param_name}.jpg', dpi=300)
            plt.close(fig)

    print('==== STORED ALL FIGURES TO HDD ====')


def plot_tau_fitting_params():
    import seaborn as sns
    from plotting_utils import hide_axis_spines
    fit_params = pd.read_csv(f'{Config.BASE_DIR}/data/cirf_fitted_parameters.csv')
    # Thresholding by R2
    # sim_r2_thresh = sim_params['R2'].quantile(0.4)  # Keep top 60% fits
    sim_r2_thresh = 0.8
    good_params = fit_params[fit_params['R2'] >= sim_r2_thresh].reset_index(drop=True)

    tau_rise_m = good_params['tau_rise'].mean()
    tau_rise_sd = good_params['tau_rise'].std()
    tau_decay_m = good_params['tau_decay'].mean()
    tau_decay_sd = good_params['tau_decay'].std()
    print(f'tau rise: {tau_rise_m:.2f} +- {tau_rise_sd:.2f} s')
    print(f'tau decay: {tau_decay_m:.2f} +- {tau_decay_sd:.2f} s')

    i = 0
    for param_name in ['tau_rise', 'tau_decay', 'R2']:
        i += 1
        if param_name != 'R2':
            param = good_params[param_name]
        else:
            param = fit_params[param_name]
        fig, ax = plt.subplots()
        sns.histplot(param, kde=True)
        plt.axvline(param.mean(), color='red', linestyle='--', linewidth=2)
        ax.axvline(param.median(), color='green', linestyle='--', linewidth=2)
        ax.text(0.4, 0.8, f'n={param.shape[0]}', transform=ax.transAxes)
        ax.text(0.4, 0.75, f'median={param.median():.2f}', transform=ax.transAxes, color='green')
        ax.text(0.4, 0.7, f'mean={param.mean():.2f} +- {param.std():.2f}', transform=ax.transAxes, color='red')

        if param_name != 'R2':
            ax.set_xlim(0, 20)
        else:
            ax.set_xlim(0, 1)
        ax.set_title(f'Distribution of {param_name}')
        ax.set_xlabel(f'{param_name} [s]')
        ax.set_ylabel('Count')
        hide_axis_spines(ax, left=True, right=False, top=False, bottom=True)
        plt.savefig(f'{Config.BASE_DIR}/figures/tau_fitting/data_{param_name}.jpg', dpi=300)
        plt.close(fig)

    print('==== STORED ALL FIGURES TO HDD ====')


def main():
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    data = ca_df_f.to_numpy()

    # RUN FITTING ON SIMULATED CA DATA
    # run_simulation(ca_sampling_rate)
    # plot_simulation()
    # exit()

    # # FIND OPTIMAL CIRF PARAMETERS BY FITTING TO DATA
    # params_df = find_optimal_cirf_parallel(data, ca_sampling_rate, show_plot=False)
    # params_df.to_csv(f'{Config.BASE_DIR}/data/cirf_fitted_parameters.csv', index=False)
    # #
    # # Remove low R2
    # r2_thresh = params_df['R2'].quantile(0.2)  # Keep top 80% fits
    # params_df_final = params_df[params_df['R2'] >= r2_thresh].reset_index(drop=True)
    # params_df_final.to_csv(f'{Config.BASE_DIR}/data/cirf_fitted_parameters_good_fits.csv', index=False)

    plot_tau_fitting_params()

    # best_parameters = params_df_final.mean()
    # best_parameters_sd = params_df_final.std()
    #
    # import seaborn as sns
    # for param in ['amplitude', 'tau_rise', 'tau_decay', 't0']:
    #     plt.figure()
    #     sns.histplot(params_df_final[param], kde=True)
    #     plt.title(f'Distribution of {param} (Good Fits)')
    #     plt.xlabel(param)
    #     plt.ylabel('Count')
    #     plt.show()


if __name__ == '__main__':
    main()
