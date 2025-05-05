import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from utils import align_traces_via_padding_df, load_hdf5_as_dict, get_rois_per_sweep
from IPython import embed


def create_alternating_pattern(group_numbers):
    alternating = []
    current_value = 0  # Start with 0
    alternating.append(current_value)

    for i in range(1, len(group_numbers)):
        if group_numbers[i] != group_numbers[i - 1]:
            current_value = 1 - current_value  # Flip between 0 and 1
        alternating.append(current_value)

    return alternating


def plot_heat_map(ca_traces, time_vector, stimulus_trace, stimulus_onsets, vr_events, sw_rois, ca_labels, file_name):

    number_of_rois = ca_traces.shape[1]
    fish_ids = create_alternating_pattern(ca_labels.iloc[0, :].to_numpy().astype('int'))
    # Create 2D array for the side bar
    fish_ids_bar = np.array(fish_ids).reshape(-1, 1)  # Shape: (neurons, 1)
    z_plane_bar = np.array(ca_labels.iloc[4, :]).reshape(-1, 1).astype('int')

    # Black and white colormap
    fish_ids_cmap = ListedColormap(['black', 'grey'])
    z_plane_cmap = ListedColormap(['red', 'blue', 'green', 'magenta', 'cyan', 'orange'])

    # Figure
    fig_width_cm = 15
    fig_height_cm = 10
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.85, wspace=0, hspace=0)

    # Create a grid that has mm resolution
    grid = matplotlib.gridspec.GridSpec(nrows=int(fig_height_cm * 10), ncols=int(fig_width_cm * 10))
    axs = dict()

    # grid[HEIGHT, WIDTH]
    # HEIGHT starts at top of page, WIDTH starts left
    # Stimulus Trace
    axs['stimulus'] = plt.subplot(grid[0:10, 0:142])

    # Response Heatmap
    axs['ca'] = plt.subplot(grid[12:100, 0:142])

    # Colorbar
    axs['cb'] = plt.subplot(grid[25+25:100-25, 148:150])

    # STIMULUS PLOT
    axs['stimulus'].plot(time_vector, stimulus_trace, 'k', lw=1)
    axs['stimulus'].set_xlim([0, time_vector[-1]])
    axs['stimulus'].set_axis_off()

    # HEATMAP
    v_min = 0
    v_max = 3
    cmap = 'afmhot'
    # axs['ca'].pcolormesh(time_vector, y, ca_traces, vmin=v_min, vmax=v_max, cmap=cmap, rasterized=True)
    axs['ca'].imshow(
        ca_traces.T, aspect='auto', extent=(time_vector[0], time_vector[-1], 0, number_of_rois),
        cmap=cmap, origin='lower', vmin=v_min, vmax=v_max
    )

    # Add Fish ID Bar
    # [x0, y0, width, height]
    inset_ax = axs['ca'].inset_axes([1.005, 0, 0.01, 1], transform=axs['ca'].transAxes)
    inset_ax.imshow(fish_ids_bar, aspect='auto', cmap=fish_ids_cmap, origin='lower')
    inset_ax.axis('off')

    # Add Z-Plane Bar
    # [x0, y0, width, height]
    inset_ax = axs['ca'].inset_axes([1.015, 0, 0.01, 1], transform=axs['ca'].transAxes)
    inset_ax.imshow(z_plane_bar, aspect='auto', cmap=z_plane_cmap, origin='lower')
    inset_ax.axis('off')

    axs['ca'].set_xlim([0, time_vector[-1]])
    axs['ca'].set_ylim([0.5, number_of_rois+0.5])
    axs['ca'].set_xlabel('Time [s]')
    axs['ca'].set_ylabel(f'Neurons')
    # axs['ca'].set_ylabel(f'ROIs (n={number_of_rois})')
    # axs['ca'].set_yticks([])

    # Plot Stimulus Onsets as a line in the heat map
    for onset in stimulus_onsets.iloc[0]:
        on_time = np.floor(onset)
        axs['ca'].plot([on_time, on_time], [0, number_of_rois + 0.5], 'g:', lw=1)

    # Plot VR Events on heat map
    # Loop over sweeps
    if vr_events is not None:
        start_roi = 0
        for sw in vr_events:
            # Get the Motor Events for this sweep
            motor_onsets = vr_events[sw]['start_time'].round()
            rois_range = len(sw_rois[sw])
            end_roi = start_roi + rois_range
            # Loop over motor events
            for m_onset in motor_onsets:
                axs['ca'].plot([m_onset, m_onset], [start_roi, end_roi], 'b', lw=1, alpha=0.75)
            start_roi = end_roi

    # COLORBAR
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cb = matplotlib.colorbar.ColorbarBase(axs['cb'], norm=norm, cmap=cmap)
    cb.ax.tick_params(labelsize=10)
    cb.ax.text(6.0, 0.5, 'dF/F', transform=cb.ax.transAxes, rotation=270, ha='center', va='center')
    cb.set_ticks([0, 1, 2, 3])

    # Store Figure to HDD
    plt.savefig(f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/{file_name}.jpg', dpi=600)
    plt.savefig(f'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/{file_name}.pdf', dpi=600)
    plt.close(fig)


def check_stimulus_delays(stimulus_protocols, sweeps, ca_sampling_rate):
    t0 = stimulus_protocols[0].iloc[1, 4].item()
    delays = []
    print(f'Reference Sweep: sw_01')
    print(f'Sampling Rate: {ca_sampling_rate} Hz --> dt = {1/ca_sampling_rate} s')
    for k, sw_protocol in enumerate(stimulus_protocols):
        sw = sw_protocol.iloc[0, 1]
        if sw in sweeps:
            time_shift = sw_protocol.iloc[1, 4].item() - t0
            if time_shift > 1/ca_sampling_rate:
                print(f'{sw}: {time_shift} s')
                delays.append(sw)
    print(f'Sweeps with Stimulus Delays larger than Sampling Rate: {len(delays)}')
    return delays


def align_sweeps_stimulus_times(stimulus_protocols, ca_labels, sweeps, ca_df_f, ca_sampling_rate):
    t0 = stimulus_protocols[0].iloc[1, 4].item()  # sweep_01
    ca_time_shifts = pd.DataFrame()
    for k, sw_protocol in enumerate(stimulus_protocols):
        sw = sw_protocol.iloc[0, 1]
        # print(f'{sw}: {sw_protocol.iloc[1, 4].item() - t0}')
        if sw in sweeps:
            time_shift = sw_protocol.iloc[1, 4].item() - t0
            print(f'{sw}: {time_shift} s')
            idx = ca_labels.columns[ca_labels.isin([sw]).any()].tolist()
            sw_delays = pd.DataFrame(columns=idx)
            sw_delays.loc[0] = [time_shift] * len(sw_delays.columns)
            ca_time_shifts = pd.concat([ca_time_shifts, sw_delays], axis=1)
        else:
            continue

    aligned_ca_df_f = align_traces_via_padding_df(ca_df_f, ca_time_shifts, ca_sampling_rate)
    return aligned_ca_df_f


def load_data():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_events_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_events.h5'

    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'
    ca_df_f_file_no_motor = f'{base_dir}/data/df_f_data_no_motor.csv'

    ca_labels_file = f'{base_dir}/data/df_f_data_labels.csv'
    stimulus_protocols_dir = f'{base_dir}/protocols'
    stimulus_trace_dir = f'{base_dir}/stimulus_traces/stimulus_traces.csv'
    stimulus_onsets_dir = f'{base_dir}/stimulus_traces/sw_01_stimulus_onsets.csv'
    # stimulus_offsets_dir = f'{base_dir}/stimulus_traces/sw_01_stimulus_offsets.csv'

    # Load data for selected sweeps
    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_df_f_no_motor = pd.read_csv(ca_df_f_file_no_motor)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()
    ca_time_axis = np.linspace(0, ca_df_f.shape[0] / ca_sampling_rate, ca_df_f.shape[0])

    # Load labels for selected sweeps
    ca_labels = pd.read_csv(ca_labels_file)

    # Load Stimulus
    stimulus_traces = pd.read_csv(stimulus_trace_dir)
    stimulus_trace = stimulus_traces['sw_01']
    stimulus_onsets = pd.read_csv(stimulus_onsets_dir, index_col=0)
    # stimulus_offsets = pd.read_csv(stimulus_offsets_dir, index_col=0)

    # Detected VR Events for selected data set
    vr_events = load_hdf5_as_dict(vr_events_dir)
    sw_rois = get_rois_per_sweep(ca_labels)

    # Check if there is any delay in stimulus times between the sweeps that is larger than the sampling rate
    # delays = check_stimulus_delays(stimulus_protocols, sweeps, ca_sampling_rate)

    # Align Stimulus Times if necessary
    # aligned_ca_df_f = align_sweeps_stimulus_times(stimulus_protocols, ca_labels, sweeps, ca_df_f, ca_sampling_rate)
    # stimulus_binary.sum(axis=1)

    return ca_df_f, ca_df_f_no_motor, ca_time_axis, stimulus_trace, stimulus_onsets, vr_events, sw_rois, ca_labels


def main():
    # Load Data
    ca_df_f, ca_df_f_no_motor, ca_time_axis, stimulus_trace, stimulus_onsets, vr_events, sw_rois, ca_labels = load_data()

    # Plot Heat Map
    # file_name = 'heatmap'
    # plot_heat_map(ca_df_f, ca_time_axis, stimulus_trace, stimulus_onsets, None, sw_rois, ca_labels, file_name)
    #
    # file_name = 'heatmap_with_vr_events'
    # plot_heat_map(ca_df_f, ca_time_axis, stimulus_trace, stimulus_onsets, vr_events, sw_rois, ca_labels, file_name)
    #
    file_name = 'heatmap_motor_corrected_reg'
    plot_heat_map(ca_df_f_no_motor, ca_time_axis, stimulus_trace, stimulus_onsets, None, sw_rois, ca_labels, file_name)

    file_name = 'heatmap_with_vr_events_motor_corrected_reg'
    plot_heat_map(ca_df_f_no_motor, ca_time_axis, stimulus_trace, stimulus_onsets, vr_events, sw_rois, ca_labels, file_name)

    print('=== STORED HEATMAPS TO HDD ====')


if __name__ == '__main__':
    main()
