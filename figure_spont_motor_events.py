from config import Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from utils import load_hdf5_as_dict, save_dict_as_hdf5, get_rois_per_sweep, norm_min_max, z_transform
import plotting_utils as pu
from IPython import embed


def plot_spontaneous_motor_responses(ca_responses, ca_responses_base_line_corrected, ca_sampling_rate):
    # Compute Time Axis
    time_vector = np.linspace(0, ca_responses.shape[0] / ca_sampling_rate, ca_responses.shape[0]) - 5

    cmap = 'afmhot'
    number_responses = ca_responses.shape[1]

    fig_width_cm = 10
    fig_height_cm = 5
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.3, right=0.9, wspace=0.0, hspace=0.0)

    # Create a grid that has mm resolution
    # grid[HEIGHT, WIDTH]
    # HEIGHT starts at top of page, WIDTH starts left
    grid = matplotlib.gridspec.GridSpec(nrows=int(fig_height_cm * 10), ncols=int(fig_width_cm * 10))

    ax1 = plt.subplot(grid[0:50, 0:15])
    # ax1_cb = plt.subplot(grid[10:40, 16:20])
    ax2 = plt.subplot(grid[0:50, 18:33])
    ax2_cb = plt.subplot(grid[10:40, 34:36])
    ax3 = plt.subplot(grid[0:50, 62:100])

    ax1.imshow(
        ca_responses.T, aspect='auto', extent=(time_vector[0], time_vector[-1], 0, number_responses),
        cmap=cmap, origin='lower', vmin=0, vmax=3
    )
    ax1.plot([0, 0], [0, number_responses], color='tab:green', lw=0.5, ls=':')
    ax1.set_xlim([time_vector[0], time_vector[-1]])
    ax1.set_ylim([0.5, number_responses+0.5])
    ax1.set_ylabel(f'Responses (n={number_responses})')
    pu.hide_axis_spines(ax1, left=False, right=False, top=False, bottom=False)

    number_responses = ca_responses_base_line_corrected.shape[1]
    ax2.imshow(
        ca_responses_base_line_corrected.T, aspect='auto', extent=(time_vector[0], time_vector[-1], 0, number_responses),
        cmap=cmap, origin='lower', vmin=0, vmax=3
    )
    ax2.plot([0, 0], [0, number_responses], color='tab:green', lw=0.5, ls=':')
    ax2.set_xlim([time_vector[0], time_vector[-1]])
    ax2.set_ylim([0.5, number_responses+0.5])
    pu.hide_axis_spines(ax2, left=False, right=False, top=False, bottom=False)
    pu.remove_y_axis(ax2, ticks=True, label=True, complete=False)
    pu.add_colorbar(
        ax2_cb, label='dF/F', label_font_size=10, label_rotation=270, label_pad=10,
        cmap=cmap, v_min=0, v_max=3, ticks=[0, 1, 2, 3]
    )
    # Add common x-axis label
    fig.text(0.33, 0.1, 'Time [s]', ha='center', fontsize=10)
    fig.text(0.78, 0.1, 'Time [s]', ha='center', fontsize=10)

    mean_response = ca_responses_base_line_corrected.mean(axis=1)
    sd_response = ca_responses_base_line_corrected.std(axis=1)
    sem_response = ca_responses_base_line_corrected.sem(axis=1)

    ax3.plot([0, 0], [-5, 15], color=(0.8, 0.8, 0.8, 1.0))
    ax3.plot(time_vector, mean_response, 'k')
    ax3.fill_between(time_vector, y1=mean_response+sem_response, y2=mean_response-sem_response, color='gray', alpha=0.3)
    ax3.set_ylim(0, 0.8)
    ax3.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax3.set_xlim(-5, 15)
    ax3.set_xticks([-5, 0, 5, 10, 15])
    # ax3.set_xlabel('Time [s]')
    # ax3.set_ylabel('dF/F')
    pu.hide_axis_spines(ax3)

    # for col in ca_responses:
    #     trace = ca_responses_base_line_corrected[col]
    #     ax3.plot(time_vector, trace, color=(0.9, 0.9, 0.9, 1))

    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/spontaneous_motor_events.jpg', dpi=600)
    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/spontaneous_motor_events.pdf', dpi=600)
    plt.close(fig)
    # plt.show()


def get_spontaneous_events(vr_events):
    sp_events = dict()
    for sw in vr_events:
        idx = vr_events[sw]['stimulus'] == 0
        sp_events[sw] = vr_events[sw][idx]
    return sp_events


def get_ca_responses(sp_events, rois_per_sweep, ca_df_f, ca_sampling_rate):
    time_before = 10  # in secs
    time_after = 20  # in secs

    idx_before = int(time_before * ca_sampling_rate)
    idx_after = int(time_after * ca_sampling_rate)

    responses = list()
    for sw in sp_events:
        sp_starts = sp_events[sw]['start_idx']
        sw_rois = rois_per_sweep[sw].astype('str')
        ca_traces = ca_df_f[sw_rois].to_numpy()

        for s_idx in sp_starts:
            cut_outs = ca_traces[s_idx-idx_before:s_idx+idx_after, :]
            if cut_outs.shape[0] == idx_before + idx_after:
                responses.append(cut_outs)

                # Check if ROIs show sign. response
                # normalize to baseline (before swim onset)
                baselines = cut_outs[0:10, :]
                baseline_mean = np.mean(baselines, axis=0)
                baseline_sd = np.std(baselines, axis=0)

                response_z = (cut_outs - baseline_mean) / baseline_sd

                embed()
                exit()

    final_df = pd.DataFrame(np.hstack(responses))
    return final_df


def create_data_for_plotting():
    # Load Data
    ca_df_f = pd.read_csv(Config.ca_df_f_file)
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
    ca_labels = pd.read_csv(Config.ca_labels_file)

    vr_events = load_hdf5_as_dict(Config.vr_events_file)
    sp_events = get_spontaneous_events(vr_events)
    save_dict_as_hdf5(Config.vr_spontaneous_events_file, sp_events)

    rois_per_sweep = get_rois_per_sweep(ca_labels)

    # Compute z-scores:
    ca_traces_z_score = z_transform(ca_df_f)

    # Get all response traces of all ROIs to all spontaneous motor events
    # (number of samples over time, number of responses)
    ca_responses = get_ca_responses(sp_events, rois_per_sweep, ca_traces_z_score, ca_sampling_rate)

    # Store to HDD
    ca_responses.to_csv(Config.ca_df_f_sp_motor_file, index=False)

    # Subtract baseline activity
    ca_responses_bl = ca_responses - ca_responses.iloc[0:10, :].mean()

    # Sort responses by strongest response (around motor event onset)
    sort_idx = ca_responses_bl.iloc[12:15, :].max(axis=0).sort_values(ascending=True).index
    ca_responses_bl_sorted = ca_responses_bl[sort_idx]
    ca_responses_sorted = ca_responses[sort_idx]

    # Store Data to HDD for later plotting

    ca_responses_sorted.to_csv(Config.ca_responses_spontaneous_motor, index=False)
    ca_responses_bl_sorted.to_csv(Config.ca_responses_spontaneous_motor_baseline_corrected, index=False)

    print('==== STORED DATA FOR FIGURE SPONTANEOUS MOTOR EVENTS TO HDD ====')


def load_data():
    ca_responses_sorted = pd.read_csv(Config.ca_responses_spontaneous_motor)
    ca_responses_bl_sorted = pd.read_csv(Config.ca_responses_spontaneous_motor_baseline_corrected)

    return ca_responses_sorted, ca_responses_bl_sorted


def main():
    create_data_for_plotting()
    exit()

    # Load Data for Plotting
    ca_responses_sorted, ca_responses_bl_sorted = load_data()
    ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()

    plot_spontaneous_motor_responses(ca_responses_sorted, ca_responses_bl_sorted, ca_sampling_rate)


if __name__ == '__main__':
    main()

