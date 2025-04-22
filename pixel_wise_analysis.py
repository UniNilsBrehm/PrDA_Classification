#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comment
"""
from config import Config
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from IPython import embed
from utils import calcium_impulse_response, create_regressors_from_binary, norm_min_max, load_hdf5_as_dict, load_tiff_recording
from skimage.exposure import rescale_intensity
import statsmodels.api as sm
from scipy.stats import linregress
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import joblib
from read_roi import read_roi_zip


# --- Helper to link tqdm with joblib ---
@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def merge_events_closing(trace, max_gap):
    from scipy.ndimage import binary_closing

    trace = np.array(trace)
    # Create a structuring element that covers gaps of size max_gap
    structure = np.ones(max_gap * 2 + 1)

    # binary_closing will fill gaps that are smaller than the structuring element
    closed = binary_closing(trace, structure=structure)
    return closed.astype(int)


def compute_df_f(trace):
    fbs = np.percentile(trace, 10)
    df_f = (trace - fbs) / fbs
    return df_f


def linear_model(trace, reg):
    # Add y-intercept
    design_matrix = sm.add_constant(reg)

    # Compute LM-OLS Model
    model = sm.OLS(trace, design_matrix).fit()
    r2 = float(model.rsquared)
    cf = float(model.params[1])
    score = cf * r2

    result = {'r2': r2, 'cf': cf, 'score': score}
    return result


def normalize_image(img, dtype=np.float32):
    """
    Normalize any image array to range [0, 1] using its own min and max values.

    Parameters:
        img (ndarray): The image to normalize.
        dtype (type): Output data type (default: float32).

    Returns:
        ndarray: Normalized image.
    """
    img = img.astype(np.float32)
    return rescale_intensity(img, in_range='image', out_range=(0.0, 1.0)).astype(dtype)


def compute_pixel_r2(signal, regressor):
    if np.any(np.isnan(signal)) or np.std(signal) == 0:
        return np.nan
    slope, intercept, r_value, p_value, std_err = linregress(regressor, signal)
    return r_value ** 2, slope


def compute_chunk_correlation(start, end, z_data, frames):
    batch = z_data[:, start:end]  # shape: (frames, batch_size)
    corr = np.dot(batch.T, z_data) / frames  # (batch_size, n_pixels)

    # Zero-out self correlations
    for i in range(start, end):
        corr[i - start, i] = np.nan

    return np.nanmean(corr, axis=1)


def compute_global_correlation_map_chunked_parallel(data, batch_size=500, n_jobs=-1):
    frames, x, y = data.shape
    n_pixels = x * y
    data_reshaped = data.reshape(frames, -1)

    # Z-score
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    z_data = (data_reshaped - data_mean) / data_std

    chunks = [(start, min(start + batch_size, n_pixels)) for start in range(0, n_pixels, batch_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_chunk_correlation)(start, end, z_data, frames)
        for start, end in tqdm(chunks, desc="Computing parallel chunks")
    )

    result_flat = np.concatenate(results)
    return result_flat.reshape(x, y)


def pixelwise_regression_parallel(data, regressor, n_jobs=-1):
    frames, x_dim, y_dim = data.shape
    data_reshaped = data.reshape(frames, -1)  # shape: (frames, x*y)

    with tqdm_joblib(tqdm(desc="Computing pixelwise regression", total=data_reshaped.shape[1])) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_pixel_r2)(data_reshaped[:, i], regressor)
            for i in range(data_reshaped.shape[1])
        )

    # Unpack results
    r2_flat, slope_flat = zip(*results)

    r2_map = np.array(r2_flat).reshape(x_dim, y_dim)
    slope_map = np.array(slope_flat).reshape(x_dim, y_dim)

    return r2_map, slope_map


def px_wise_linear_scoring(rec, binaries):
    r2, slope, score = dict(), dict(), dict()
    _, cirf = calcium_impulse_response(tau_rise=3, tau_decay=6, norm=True, sampling_rate=3)
    for s_type in binaries:
        print(f'==== {s_type} ====')
        binary = binaries[s_type]
        if binary.any():
            reg = create_regressors_from_binary(binary, cirf)
            r2_map, slope_map = pixelwise_regression_parallel(rec, reg)
            score_map = r2_map * slope_map
            r2[s_type] = r2_map
            slope[s_type] = slope_map
            score[s_type] = score_map
        else:
            zero_image = np.zeros((rec.shape[1], rec.shape[2]))
            r2[s_type] = zero_image
            slope[s_type] = zero_image
            score[s_type] = zero_image

    results = {'r2': r2, 'slope': slope, 'score': score}
    return results


def plot_r2_maps_grid(data_dict, save_path, stimulus_types, cmap='hot', v_min=0, v_max=1, figsize_per_plot=(3, 3)):
    """s
    Plots a dictionary of 2D arrays as a grid of heatmaps.

    Parameters:
        data_dict (dict): Dictionary where keys are labels and values are 2D numpy arrays.
        save_path (str): File path to save the final combined figure.
        cmap (str): Colormap for heatmaps.
        v_min (float): Minimum value for colormap.
        v_max (float): Maximum value for colormap.
        figsize_per_plot (tuple): Size (width, height) of each subplot.
    """
    title_text = os.path.split(save_path)[1]
    n = len(data_dict)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
    fig.suptitle(title_text, fontsize=14)
    axs = axs.flatten()
    # for i, (k, data_map) in enumerate(data_dict.items()):
    for i, k in enumerate(stimulus_types):
        data_map = data_dict[k]
        im = axs[i].imshow(data_map, cmap=cmap, vmin=v_min, vmax=v_max)
        axs[i].set_title(f'{k} R2')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def get_stimulus_binaries(file_dir):
    stimulus_binaries = pd.read_csv(file_dir, index_col=0)
    stimulus_binaries['moving_target'] = stimulus_binaries['moving_target_01'] + stimulus_binaries['moving_target_02']
    stimulus_binaries = stimulus_binaries.drop(columns=['moving_target_01', 'moving_target_02'])
    return stimulus_binaries


def plot_imagej_rois(ax, rois, show_label=False):
    for idx, (name, roi) in enumerate(rois.items(), start=1):
        if roi.get('type') != 'oval':
            continue  # Skip anything that's not an oval

        x_center = roi['left'] + roi['width'] / 2
        y_center = roi['top'] + roi['height'] / 2

        ellipse = plt.matplotlib.patches.Ellipse(
            (x_center, y_center),
            roi['width'],
            roi['height'],
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )
        ax.add_patch(ellipse)

        if show_label:
            # Add index number in the center
            ax.text(x_center, y_center, str(idx),
                    color='red', fontsize=8, ha='center', va='center')


def plot_heat_maps(r2_map, slope_map, score_map, im, rois, save_dir):
    title_text = os.path.split(save_dir)[1]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title_text, fontsize=14)
    im0 = axs[0, 0].imshow(im, cmap='gray', vmin=im.min(), vmax=im.max() / 2)
    if rois is not None:
        plot_imagej_rois(axs[0, 0], rois, show_label=True)
    im1 = axs[0, 1].imshow(r2_map, cmap='hot', vmin=r2_map.min(), vmax=r2_map.max())
    im2 = axs[1, 0].imshow(slope_map, cmap='hot', vmin=np.percentile(slope_map, 5), vmax=slope_map.max() / 2)
    im3 = axs[1, 1].imshow(score_map, cmap='hot', vmin=np.percentile(score_map, 5), vmax=score_map.max() / 2)
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04, shrink=0.5)
    axs[0, 0].set_title('Mean Image')
    axs[0, 1].set_title('R squared')
    axs[1, 0].set_title('Slope')
    axs[1, 1].set_title('Score')
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.close(fig)


def plot_heat_maps_fixed_scale(r2_map, slope_map, score_map, im, v_min, v_max, save_dir, cmap='hot'):
    title_text = os.path.split(save_dir)[1]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title_text, fontsize=14)
    im0 = axs[0, 0].imshow(im, cmap='gray', vmin=im.min(), vmax=im.max() / 2)
    im1 = axs[0, 1].imshow(r2_map, cmap=cmap, vmin=v_min['r2'], vmax=v_max['r2'])
    im2 = axs[1, 0].imshow(slope_map, cmap=cmap, vmin=v_min['slope'], vmax=v_max['slope'])
    im3 = axs[1, 1].imshow(score_map, cmap=cmap, vmin=v_min['score'], vmax=v_max['score'])
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04, shrink=0.5)
    axs[0, 0].set_title('Mean Image')
    axs[0, 1].set_title('R squared')
    axs[1, 0].set_title('Slope')
    axs[1, 1].set_title('Score')
    plt.tight_layout()
    plt.savefig(save_dir, dpi=600)
    plt.close(fig)


def plot_single_heat_maps_fixed_scale(v_map, v_min, v_max, save_dir, cmap='hot'):
    fig, axs = plt.subplots(figsize=(5, 5))
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    axs.imshow(v_map, cmap=cmap, vmin=v_min, vmax=v_max)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def linear_scoring_pixel_wise(rec, reg):
    # _, cirf = calcium_impulse_response(tau_rise=3, tau_decay=6, norm=True, sampling_rate=3)
    r2_map, slope_map = pixelwise_regression_parallel(rec, reg)
    score_map = r2_map * slope_map
    return r2_map, slope_map, score_map


def linear_scoring_px_mean_activity_map(tiff_file, save_dir):
    tif_rec = load_tiff_recording(tiff_file, flatten=True)
    rec = normalize_image(tif_rec)

    # Linear Scoring with Mean Activity
    mean_trace = norm_min_max(np.mean(rec, axis=(1, 2)))
    mean_image = norm_min_max(np.mean(rec, axis=0))
    mean_r2_map, mean_slope_map, mean_score_map = linear_scoring_pixel_wise(rec, mean_trace)

    v_min = {'r2': 0, 'slope': 0, 'score': 0}
    v_max = {'r2': 0.3, 'slope': 0.3, 'score': 0.02}

    # Store mean activity heatmaps
    f_name = f'{save_dir}/px_mean_activity_maps'
    plot_heat_maps_fixed_scale(
        mean_r2_map,
        mean_slope_map,
        mean_score_map,
        mean_image,
        v_min=v_min,
        v_max=v_max,
        save_dir=f_name,
        cmap='gray'
    )

    f = f'{save_dir}/px_r2_mean_activity_map'
    plot_single_heat_maps_fixed_scale(mean_r2_map, v_min['r2'], v_max['r2'], save_dir=f, cmap='gray')

    f = f'{save_dir}/px_slope_mean_activity_map'
    plot_single_heat_maps_fixed_scale(mean_slope_map, v_min['slope'], v_max['slope'], save_dir=f, cmap='gray')

    f = f'{save_dir}/px_score_mean_activity_map'
    plot_single_heat_maps_fixed_scale(mean_score_map, v_min['score'], v_max['score'], save_dir=f, cmap='gray')


def batch_mean_activity(base_dir):
    import time
    sw_list = os.listdir(base_dir)
    k = 0

    for sw in sw_list:
        t0 = time.perf_counter()
        k += 1
        sw_dir = f'{base_dir}/{sw}'
        rec_dir = f'{sw_dir}/rec'
        tif_file = os.listdir(rec_dir)
        if len(tif_file) == 0:
            print(f'\n==== ERROR: TIFF FILE RECORDING NOT FOUND! ({sw})=====\n')
            continue
        else:
            tif_dir = f'{rec_dir}/{tif_file[0]}'
            output_folder = f'{sw_dir}/px_linear_scoring'
            os.makedirs(output_folder, exist_ok=True)
            linear_scoring_px_mean_activity_map(tif_dir, output_folder)

            t1 = time.perf_counter()
            print(f'FINISHED {k}/{len(sw_list)}, this took {(t1 - t0)/60:.3f} minutes')


def roi_detection():
    tif_file = 'D:/WorkingData/RoiDetection/test/rec/sw_25_motion_corrected.tif'
    ca_sampling_rate = 3
    tif_rec = load_tiff_recording(tif_file, flatten=True)
    rec = normalize_image(tif_rec)

    # Linear Scoring with Mean Activity
    mean_trace = norm_min_max(np.mean(rec, axis=(1, 2)))
    mean_image = norm_min_max(np.mean(rec, axis=0))
    mean_r2_map, mean_slope_map, mean_score_map = linear_scoring_pixel_wise(rec, mean_trace)

    # Manual Binary
    # plt.plot(mean_trace, 'k')
    # plt.show()

    binary = np.zeros_like(mean_trace)
    binary[[1090, 1240, 1420, 1600, 1750, 1900, 2060, 2215]] = 1
    _, cirf = calcium_impulse_response(tau_rise=3, tau_decay=6, norm=True, sampling_rate=3)
    reg = create_regressors_from_binary(binary, cirf, norm=True)
    b_r2_map, b_slope_map, b_score_map = linear_scoring_pixel_wise(rec, reg)

    residuals = mean_trace - reg
    residuals[residuals <= 0] = np.percentile(residuals, 50) + (0.01 * np.random.randn(len(residuals[residuals <= 0])))
    res_r2_map, res_slope_map, res_score_map = linear_scoring_pixel_wise(rec, residuals)

    # plt.plot(norm_min_max(mean_trace), 'k')
    # plt.plot(reg, 'g')
    # plt.plot(residuals, 'r')
    # plt.show()

    save_dir = f'D:/WorkingData/RoiDetection/test/rec/px_regression'
    v_min = {'r2': 0, 'slope': 0, 'score': 0}
    v_max = {'r2': 0.5, 'slope': 0.1, 'score': 0.1}

    # Store mean activity heatmaps
    f_name = f'{save_dir}/mean_activity_maps'
    plot_heat_maps_fixed_scale(mean_r2_map, mean_slope_map, mean_score_map, mean_image, v_min=v_min, v_max=v_max, save_dir=f_name)

    # Store Manual Binary heatmaps
    f_name = f'{save_dir}/b_activity_maps'
    plot_heat_maps_fixed_scale(b_r2_map, b_slope_map, b_score_map, mean_image, v_min=v_min, v_max=v_max, save_dir=f_name)

    # Store Manual Binary Residuals heatmaps
    f_name = f'{save_dir}/residuals_activity_maps'
    plot_heat_maps_fixed_scale(res_r2_map, res_slope_map, res_score_map, mean_image, v_min=v_min, v_max=v_max, save_dir=f_name)

    embed()
    exit()
    # Save as TIFF stack
    # Normalize and convert to uint16 (if needed)
    corrected_array = mean_r2_map - mean_r2_map.min()
    corrected_array /= corrected_array.max()
    corrected_array = (corrected_array * 65535).astype(np.uint16)
    output_path = f'D:/WorkingData/RoiDetection/test/mean_r2_map.tif'
    tiff.imwrite(
        output_path,
        corrected_array,
        dtype=np.uint16,
        # bigtiff=False,
        imagej=True,
        # metadata=None,
        photometric='minisblack',
        # planarconfig=None,
        # compress=None,
        # tile=None,
        # resolution=None,
        # description=None
    )

    # Plots
    # Mean Activity
    # save_dir = f'D:/WorkingData/PrTecDA_Data/Neuropil_LTe_Ca_imaging/figures/mean_activity_maps'
    save_dir = f'D:/WorkingData/RoiDetection/test/rec/mean_activity_maps'
    plot_heat_maps(mean_r2_map, mean_slope_map, mean_score_map, mean_image, rois=None, save_dir=save_dir)


def run_pixel_regression():
    # tif_file = 'D:/WorkingData/RoiDetection/test/somata.tif'
    sw = 'sw_17'
    tif_file = f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/tiff_stacks/{sw}/{sw}_ca2+_reg.tif'
    ca_sampling_rate = 2.79005402323014
    tif_rec = load_tiff_recording(tif_file, flatten=False)
    rec = normalize_image(tif_rec)
    _, cirf = calcium_impulse_response(tau_rise=3, tau_decay=6, norm=True, sampling_rate=3)

    stimulus_binaries_dir = f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/stimulus_traces/{sw}_stimulus_binaries.csv'
    stimulus_binaries = get_stimulus_binaries(stimulus_binaries_dir)

    stimulus_types = [
        'moving_target',
        'grating_appears',
        'grating_0',
        'grating_180',
        'grating_disappears',
        'bright_loom',
        'dark_loom',
        'bright_flash',
        'dark_flash'
    ]

    # Get VR Data
    vr_binaries = load_hdf5_as_dict(Config.vr_all_binaries_file)
    motor_events = vr_binaries[sw]

    # Collect Motor Events
    motor_spontaneous = motor_events['spontaneous']

    # Merge Motor Events that are too close
    max_gap_secs = 10  # in secs
    motor_spontaneous_events = merge_events_closing(motor_spontaneous, max_gap=int(max_gap_secs * ca_sampling_rate))
    # motor_spontaneous_reg = create_regressors_from_binary(motor_spontaneous_events, cirf)
    # stimulus_binaries['motor_spontaneous'] = motor_spontaneous_events

    # Get ROIs from Imagej
    roi_zip_path = f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/tiff_stacks/{sw}/{sw}_RoiSet.zip'
    rois = read_roi_zip(roi_zip_path)
    mean_image = norm_min_max(np.mean(rec, axis=0))

    # Linear Scoring with Mean Activity
    mean_trace = np.mean(rec, axis=(1, 2))
    mean_r2_map, mean_slope_map, mean_score_map = linear_scoring_pixel_wise(rec, mean_trace)

    # Linear Scoring with Spontaneous Motor
    reg = create_regressors_from_binary(motor_spontaneous_events, cirf)
    spontaneous_r2_map, spontaneous_slope_map, spontaneous_score_map = linear_scoring_pixel_wise(rec, reg)

    # Linear Scoring with Stimulus Regressors and each pixel
    res_stimulus_regs = px_wise_linear_scoring(rec, stimulus_binaries)

    # Linear Scoring with Motor Regressors and each pixel
    motor_binaries = motor_events.drop(columns=['spontaneous'])
    motor_binaries['moving_target'] = motor_binaries['moving_target_01'] + motor_binaries['moving_target_02']
    motor_binaries = motor_binaries.drop(columns=['moving_target_01', 'moving_target_02'])
    res_motor_regs = px_wise_linear_scoring(rec, motor_binaries)

    # ==================================================================================================================
    # Plots
    # Mean Activity
    save_dir = f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/test/{sw}_mean_activity_maps.jpg'
    plot_heat_maps(mean_r2_map, mean_slope_map, mean_score_map, mean_image, rois, save_dir=save_dir)

    # Spontaneous Motor
    save_dir = f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/test/{sw}_spontaneous_maps.jpg'
    plot_heat_maps(spontaneous_r2_map, spontaneous_slope_map, spontaneous_score_map, mean_image, rois, save_dir=save_dir)

    # Stimulus Regressors
    limits = {'r2': (0, 0.3), 'slope': (0, 0.2), 'score': (0, 0.001)}
    for k in res_stimulus_regs:
        plot_r2_maps_grid(
            data_dict=res_stimulus_regs[k],
            stimulus_types=stimulus_types,
            v_min=limits[k][0],
            v_max=limits[k][1],
            save_path=f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/test/{sw}_stimulus_{k}_maps.jpg'
        )

    # Stimulus Motor Regressors
    limits = {'r2': (0, 0.3), 'slope': (0, 0.2), 'score': (0, 0.001)}
    for k in res_motor_regs:
        plot_r2_maps_grid(
            data_dict=res_motor_regs[k],
            stimulus_types=stimulus_types,
            v_min=limits[k][0],
            v_max=limits[k][1],
            save_path=f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/test/{sw}_motor_{k}_maps.jpg'
        )


def main():
    data_dir = 'F:/WorkingData/Tec_Data/Neuropil_RTe_Ca_imaging/cell_detection'
    batch_mean_activity(data_dir)


if __name__ == "__main__":
    main()
