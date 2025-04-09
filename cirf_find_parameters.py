import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import calcium_impulse_response, create_regressors_from_binary, detect_peaks
from config import Config
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from IPython import embed


def shifted_double_exponential(t, A, tau_rise, tau_decay, t0):
    response = np.zeros_like(t)
    valid_idx = t >= t0
    t_shifted = t[valid_idx] - t0
    response[valid_idx] = A * (np.exp(-t_shifted / tau_decay) - np.exp(-t_shifted / tau_rise))
    return response


def gaussian_exp_decay(t, A, t0, sigma, tau_decay):
    response = np.zeros_like(t)
    rise_part = t <= t0
    decay_part = t > t0
    response[rise_part] = A * np.exp(-((t[rise_part] - t0)**2) / (2 * sigma**2))
    response[decay_part] = A * np.exp(-(t[decay_part] - t0) / tau_decay)
    return response


def asymmetric_gaussian(t, A, mu, sigma_rise, sigma_decay):
    response = np.zeros_like(t)
    rise_part = t < mu
    decay_part = t >= mu
    response[rise_part] = A * np.exp(-((t[rise_part] - mu)**2) / (2 * sigma_rise**2))
    response[decay_part] = A * np.exp(-((t[decay_part] - mu)**2) / (2 * sigma_decay**2))
    return response


def double_exponential(t, amplitude, tau_rise, tau_decay):
    response = amplitude * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))
    response[t < 0] = 0  # Optional: zero for t < 0, depending on your data
    return response


def bi_exponential_decay(t, amplitude_1, tau1, amplitude_2, tau2):
    return amplitude_1 * np.exp(-t / tau1) + amplitude_2 * np.exp(-t / tau2)


def compute_r2(data, fitted):
    # Calculate R²
    ss_res = np.sum((data - fitted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def find_optimal_cirf(ca_data, ca_sampling_rate, show_plot=False):
    double_exp_parameters = list()
    # asymmetric_gaussian_parameters = list()
    # gaussian_exp_parameters = list()
    ca_traces_collection = list()

    before = 0
    after = 60
    k = 0
    for ca_trace in ca_data.T:
        # if k >= 5:
        #     embed()
        #     exit()
        k += 1
        print(f'==== STARTING ROI: {k} / {ca_data.shape[1]} ==== ', end='\r')
        ca_peaks = find_ca_signals(ca_trace, ca_sampling_rate)

        for peak in ca_peaks:
            # CUTOUT CA TRACE
            try:
                ca_cutout = ca_trace[peak-before:peak+after]
                ca_cutout = ca_cutout - np.min(ca_cutout)
                t = np.linspace(0, ca_cutout.shape[0] / ca_sampling_rate, ca_cutout.shape[0])
                ca_traces_collection.append(ca_cutout)
                # t_fitted = np.arange(0, t.max(), 1/1000)
            except:
                continue

            # FIT CIRF FUNCTIONS TO DATA
            # Shifted Double Exp: A, tau_rise, tau_decay, t0
            try:
                bounds = ([0, 0, 0, 0], [np.inf, 100, 100, np.max(t)])
                popt1 = curve_fit(shifted_double_exponential, t, ca_cutout, p0=[1.0, 1.0, 3, int(np.max(t))/2], bounds=bounds, maxfev=5000)
                fitted_response1 = shifted_double_exponential(t, *popt1[0])

                r2 = compute_r2(ca_cutout, fitted_response1)
                double_exp_parameters.append(np.append(popt1[0], r2))

            except RuntimeError:
                print('Could not find optimal parameters')

            # # Asymmetric Gaussian: A, t0, sigma_rise, sigma_decay
            # try:
            #     bounds = ([0, 0, 0, 0], [np.inf, np.max(t), 100, 100])
            #     popt2 = curve_fit(asymmetric_gaussian, t, ca_cutout, bounds=bounds, maxfev=1000)
            #     fitted_response2 = asymmetric_gaussian(t, *popt2[0])
            #     r2 = compute_r2(ca_cutout, fitted_response2)
            #     asymmetric_gaussian_parameters.append(np.append(popt2[0], r2))
            #
            # except RuntimeError:
            #     print('Could not find optimal parameters')
            #
            # # Gaussian Rise + Exp Decay: A, t0, sigma, tau_decay
            # try:
            #     bounds = ([0, 0, 0, 0], [np.inf, np.max(t), 100, 100])
            #     popt3 = curve_fit(gaussian_exp_decay, t, ca_cutout, bounds=bounds, maxfev=1000)
            #     fitted_response3 = gaussian_exp_decay(t, *popt3[0])
            #     r2 = compute_r2(ca_cutout, fitted_response3)
            #     gaussian_exp_parameters.append(np.append(popt3[0], r2))
            # except RuntimeError:
            #     print('Could not find optimal parameters')
            # #
            if show_plot:
                plt.figure(figsize=(10, 5))
                plt.plot(t, ca_cutout, 'k', lw=4, label='Data')
                plt.plot(t, fitted_response1, 'r-', lw=2, label=f'Double Exp')
                # plt.plot(t, fitted_response2, 'g-', lw=2, label=f'Asymmetric Gauss')
                # plt.plot(t, fitted_response3, 'b-', lw=2, label=f'Gauss + Exp')

                plt.xlabel('Time (s)')
                plt.ylabel('Response')
                plt.legend()
                plt.title('Double Exponential Fit to Calcium Trace')
                plt.show()

    ca_traces_collection = pd.DataFrame(ca_traces_collection).transpose()
    # m = ca_traces_collection.mean(axis=1)
    # f = shifted_double_exponential(t, 2, 3, 7, 0)
    double_exp_parameters = pd.DataFrame(double_exp_parameters, columns=['amplitude', 'tau_rise', 'tau_decay', 't0', 'R2'])
    double_exp_parameters.to_csv(f'{Config.BASE_DIR}/data/cirf_fitted_parameters.csv', index=False)
    # asymmetric_gaussian_parameters = pd.DataFrame(asymmetric_gaussian_parameters, columns=['amplitude', 't_center', 'sigma_rise', 'sigma_decay', 'R2'])
    # gaussian_exp_parameters = pd.DataFrame(gaussian_exp_parameters, columns=['amplitude', 't_center', 'sigma_rise', 'tau_decay', 'R2'])

    # Remove low R2
    double_exp_parameters_final = double_exp_parameters[double_exp_parameters['R2'] >= 0.8].reset_index(drop=True)
    double_exp_parameters_final.to_csv(f'{Config.BASE_DIR}/data/cirf_fitted_parameters_good_fits.csv', index=False)
    # final2 = asymmetric_gaussian_parameters[asymmetric_gaussian_parameters['R2'] >= 0.5].reset_index(drop=True)
    # final3 = gaussian_exp_parameters[gaussian_exp_parameters['R2'] >= 0.5].reset_index(drop=True)


def find_ca_signals(ca_trace, ca_sampling_rate):
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

    # corr_trace = corr[center:n + center]
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(cif, 'g')
    # axs[0].plot(ca_trace, 'k')
    # axs[0].plot(ca_peaks, ca_trace[ca_peaks], 'rx')
    # axs[1].plot(corr_trace, 'k')
    # axs[1].plot(ca_peaks, corr_trace[ca_peaks], 'rx')
    # axs[1].plot([0, corr_trace.shape[0]], [corr_th, corr_th], 'r--')
    # axs[1].axhline(corr_th, color='r', linestyle='--')
    # for p in ca_peaks:
    #     axs[0].axvline(p, color='black', linestyle='--', alpha=0.5)
    #     axs[1].axvline(p, color='black', linestyle='--', alpha=0.5)
    # plt.show()

    return ca_peaks


def main():
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    # ca_onsets = find_ca_signals(ca_df_f.iloc[:, 200].to_numpy(), ca_sampling_rate)
    find_optimal_cirf(ca_df_f.to_numpy(), ca_sampling_rate)


if __name__ == '__main__':
    main()
