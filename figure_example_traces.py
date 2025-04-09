import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from config import Config
from utils import get_rois_per_sweep, load_hdf5_as_dict, save_dict_as_hdf5
from plotting_utils import add_scale_bar, draw_sizebar
from utils import calcium_impulse_response, create_regressors_from_binary
import matplotlib.lines as lines
from plotting_style import PlotStyle
from IPython import embed


def select_trace_regions(x, y):
    """
    Allows the user to select multiple start and stop points on a plotted trace using Ctrl + Click.

    Parameters:
    x (array-like): X-axis data points.
    y (array-like): Y-axis data points.

    Returns:
    list of tuples: Each tuple contains (start_index, stop_index).
    """
    selected_regions = []
    selected_points = []

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Trace")
    ax.set_title("Use Ctrl + Click to select points, then close window when done.")
    ax.legend()

    def on_click(event):
        """Handles mouse click events."""
        # Only accept clicks if Ctrl is held and the left mouse button is used
        if event.button == 1 and event.key == 'control':
            selected_points.append((event.xdata, event.ydata))
            print(f"Point selected at x={event.xdata:.2f}")

            # If two points are selected, store the region
            if len(selected_points) == 2:
                start, stop = selected_points
                start_idx = np.abs(x - start[0]).argmin()
                stop_idx = np.abs(x - stop[0]).argmin()

                selected_regions.append((start_idx, stop_idx))
                print(f"Selected range: {start_idx} to {stop_idx}")

                # Highlight selection
                ax.axvspan(x[start_idx], x[stop_idx], color='red', alpha=0.3)
                plt.draw()

                # Reset points for the next selection
                selected_points.clear()

    # Connect event listener
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()
    return selected_regions


def plot_example_traces(data, ref_img, rois):
    stimulus_trace = data['stimulus_trace']
    stimulus_onsets = data['stimulus_onsets']
    vr_trace_org = data['vr_recording']
    vr_events = data['vr_events']
    vr_binary = data['vr_binary']
    ca_traces = data['ca_traces']
    vr_fr = float(data['vr_sampling_rate'].values[0][0])
    ca_fr = float(data['ca_sampling_rate'].values[0][0])

    # Create Time Axis
    vr_time_axis = np.linspace(0, vr_trace_org.shape[0] / vr_fr, vr_trace_org.shape[0])
    ca_time_axis = np.linspace(0, ca_traces.shape[0] / ca_fr, ca_traces.shape[0])

    # Find Artefacts manually
    # idx = select_trace_regions(vr_time_axis, vr_trace)
    # artefacts = pd.DataFrame(idx, columns=['start', 'stop'])
    # artefacts.to_csv('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/data/vr_example_trace_artefacts.csv', index=False)
    # Remove artefacts from VR Trace
    vr_trace = vr_trace_org.copy()
    # art_idx = pd.read_csv('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/data/vr_example_trace_artefacts.csv')
    # for on, off in zip(art_idx['start'], art_idx['stop']):
    #     vr_trace.iloc[on:off] = vr_trace.iloc[on:off].mean()

    # Figure
    fig_width_cm = 15
    fig_height_cm = 20
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.05, top=0.99, bottom=0.01, right=0.95, wspace=0, hspace=0)

    # Create a grid that has mm resolution
    # grid[HEIGHT, WIDTH]
    # HEIGHT starts at top of page, WIDTH starts left
    grid = matplotlib.gridspec.GridSpec(nrows=int(fig_height_cm * 10), ncols=int(fig_width_cm * 10))
    axs = dict()

    axs['ref_img'] = plt.subplot(grid[0:46, 5:70])

    # axs['stimulus'] = plt.subplot(grid[60:70, 0:75])
    # axs['ventral_root'] = plt.subplot(grid[60:100, 0:75])
    # axs['ca_traces'] = plt.subplot(grid[100:200, 0:75])

    # axs['stimulus'] = plt.subplot(grid[60:70, 0:150])
    axs['ventral_root'] = plt.subplot(grid[60:100, 0:150])
    axs['ca_traces'] = plt.subplot(grid[100:200, 0:150])

    # PLOTTING =========================================================================================================
    # Ref Image
    axs['ref_img'].imshow(ref_img)  # 260 x 192 (96 dpi, 24 bit)
    axs['ref_img'].axis('off')

    # Indicate Stimulus Onsets
    for k in stimulus_onsets:
        onset = float(stimulus_onsets[k].values[0])
        axs['ventral_root'].axvline(x=onset, color='black', linestyle='-', lw=2, alpha=0.22)
        axs['ca_traces'].axvline(x=onset, color='black', linestyle='-', lw=2, alpha=0.2)

    # STIMULUS TRACE
    # axs['stimulus'].plot(ca_time_axis, stimulus_trace, 'k', lw=1)

    # VENTRAL ROOT TRACE
    axs['ventral_root'].plot(vr_time_axis, vr_trace, 'k', lw=1)

    # CA IMAGING TRACES
    shift = 0
    for roi in rois:
        axs['ca_traces'].plot(ca_time_axis, ca_traces[roi] - shift, 'k', lw=1)
        shift += 6

    # Indicate Motor Event Onsets
    for onset in vr_events['start_time']:
        # axs['ventral_root'].plot(onset, vr_trace.max()+2, marker="v", markersize=2, markerfacecolor="tab:red", markeredgecolor="tab:red", linestyle="None")
        axs['ventral_root'].plot(onset, 15, marker="v", markersize=2, markerfacecolor="tab:red", markeredgecolor="tab:red", linestyle="None")

    # Axes
    # X Limit
    x_lim_min = 60  # secs
    x_lim_max = 960
    axs['ventral_root'].set_xlim(x_lim_min, x_lim_max)
    axs['ca_traces'].set_xlim(x_lim_min, x_lim_max)

    # Y Limit
    axs['ventral_root'].set_ylim(-10, 25)

    # Remove Axes
    # axs['stimulus'].axis('off')
    axs['ventral_root'].axis('off')
    axs['ca_traces'].axis('off')

    # Add Scale Bars
    # Time Scale Bar
    add_scale_bar(
        axs['ca_traces'], size=100, label='100 s', location=(0, 0.12), orientation='horizontal', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    # Delta F over F Scale Bar
    add_scale_bar(
        axs['ca_traces'], size=2, label='2 dF/F', location=(0.0, 0.90), orientation='vertical', color='black',
        linewidth=4, fontsize=10, padding=0.02)

    # Ventral Root mV Scale Bar
    add_scale_bar(
        axs['ventral_root'], size=5, label='5 mV', location=(0.0, 0.7), orientation='vertical', color='black',
        linewidth=4, fontsize=10, padding=0.02)

    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/example_traces.pdf', dpi=600)
    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/example_traces.jpg', dpi=600)
    plt.close(fig)
    # plt.show()
    # embed()
    # exit()


def plot_example_traces2(data, rois):
    style = PlotStyle()

    stimulus_trace = data['stimulus_trace']
    stimulus_onsets = data['stimulus_onsets']
    stimulus_binary = data['stimulus_binary']
    vr_trace = data['vr_recording']
    vr_events = data['vr_events']
    ca_traces = data['ca_traces']
    vr_fr = float(data['vr_sampling_rate'].values[0][0])
    ca_fr = float(data['ca_sampling_rate'].values[0][0])

    # Create Time Axis
    vr_time_axis = np.linspace(0, vr_trace.shape[0] / vr_fr, vr_trace.shape[0])
    ca_time_axis = np.linspace(0, ca_traces.shape[0] / ca_fr, ca_traces.shape[0])

    # Generate Calcium Impulse Response Function
    _, cif = calcium_impulse_response(tau_rise=3, tau_decay=7, amplitude=1.0,
                                      sampling_rate=ca_fr, threshold=1e-5, norm=True)

    stimulus_regs = dict()
    for s_type in stimulus_binary:
        stimulus_regs[s_type] = create_regressors_from_binary(stimulus_binary[s_type], cif, delta=True, norm=True)

    vr_reg = create_regressors_from_binary(data['vr_binary'], cif, delta=True, norm=True)

    # Figure
    fig_width_cm = 12
    fig_height_cm = 20
    fig_width = float(fig_width_cm / 2.54)
    fig_height = float(fig_height_cm / 2.54)

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.05, top=0.99, bottom=0.01, right=0.95, wspace=0, hspace=0)

    # Create a grid that has mm resolution
    # grid[HEIGHT, WIDTH]
    # HEIGHT starts at top of page, WIDTH starts left
    grid = matplotlib.gridspec.GridSpec(nrows=int(fig_height_cm * 10), ncols=int(fig_width_cm * 10))
    axs = dict()
    axs_scoring = dict()

    axs['stimulus'] = plt.subplot(grid[0:5, 0:120])
    axs['ventral_root'] = plt.subplot(grid[10:30, 0:120])
    axs['ventral_root_reg'] = plt.subplot(grid[30:40, 0:120])
    axs['stimulus_reg'] = plt.subplot(grid[45:55, 0:120])
    axs['ca_trace_0'] = plt.subplot(grid[60:80, 0:120])
    axs['ca_trace_1'] = plt.subplot(grid[80:100, 0:120])
    axs['ca_trace_2'] = plt.subplot(grid[100:120, 0:120])
    axs['ca_trace_3'] = plt.subplot(grid[120:140, 0:120])
    axs['ca_trace_4'] = plt.subplot(grid[140:160, 0:120])
    axs_scoring['example_1'] = plt.subplot(grid[165:200, 0:40])
    axs_scoring['example_2'] = plt.subplot(grid[165:200, 40:80])
    axs_scoring['example_3'] = plt.subplot(grid[165:200, 80:120])

    # axs['ca_trace_5'] = plt.subplot(grid[160:180, 0:120])
    # axs['ca_trace_6'] = plt.subplot(grid[180:200, 0:120])

    # PLOTTING =========================================================================================================
    # STIMULUS TRACE
    axs['stimulus'].plot(ca_time_axis, stimulus_trace, **style.lsStimulus)

    # VENTRAL ROOT TRACE
    axs['ventral_root'].plot(vr_time_axis, vr_trace, **style.lsVrTrace)

    # Indicate Motor Event Onsets
    for onset in vr_events['start_time']:
        axs['ventral_root'].plot(onset, 15, marker="v", markersize=2, markerfacecolor="tab:red", markeredgecolor="tab:red", linestyle="None")
        # # Add a vertical line in figure coordinates
        # line = lines.Line2D([onset, onset], [0, 1], color='red', linestyle='--')
        # fig.add_artist(line)

    # VENTRAL ROOT REG
    axs['ventral_root_reg'].plot(ca_time_axis, vr_reg, **style.lsVrReg)
    axs['ventral_root_reg'].set_ylim(-0.1, 1.1)

    # STIMULUS REG
    all_regs = np.array([sum(values) for values in zip(*stimulus_regs.values())])
    mt_reg = stimulus_regs['moving_target_01'] + stimulus_regs['moving_target_02']
    axs['stimulus_reg'].plot(ca_time_axis, all_regs, **style.lsStimulusReg_fading)
    axs['stimulus_reg'].plot(ca_time_axis, mt_reg, **style.lsStimulusReg_highlight)
    axs['stimulus_reg'].set_ylim(-0.1, 1.1)

    # CA IMAGING TRACES
    axs[f'ca_trace_0'].fill_between(ca_time_axis, all_regs * 3, y2=0, **style.fsStimulusReg)
    axs[f'ca_trace_0'].fill_between(ca_time_axis, vr_reg * 3, y2=0, **style.fsVrReg)

    for k in range(5):
        axs[f'ca_trace_{k}'].plot(ca_time_axis, ca_traces[rois[k]], **style.lsCaTrace)
        axs[f'ca_trace_{k}'].set_ylim(-1, 8)
        # axs[f'ca_trace_{k}'].fill_between(ca_time_axis, vr_reg*3, y2=0, edgecolor=None, facecolor='red', alpha=0.4)
        # axs[f'ca_trace_{k}'].fill_between(ca_time_axis, all_regs*3, y2=0, edgecolor=None, facecolor='blue', alpha=0.4)

        # for onset in vr_events['start_time']:
        #     axs[f'ca_trace_{k}'].axvline(x=onset, color='red', linestyle='-', linewidth=0.2, alpha=0.2)

    # SCORING EXAMPLES
    start = int(150 * ca_fr)
    end = int(300 * ca_fr)
    axs_scoring['example_1'].plot(ca_time_axis[start:end], ca_traces[rois[-1]][start:end], **style.lsCaTrace)
    axs_scoring['example_1'].plot(ca_time_axis[start:end]-1, vr_reg[start:end], **style.lsVrReg_spontaneous)

    start = int(480 * ca_fr)
    end = int(630 * ca_fr)
    axs_scoring['example_2'].plot(ca_time_axis[start:end], ca_traces[rois[-1]][start:end], **style.lsCaTrace)
    grating_reg = stimulus_regs['grating_0'] + stimulus_regs['grating_180']
    axs_scoring['example_2'].plot(ca_time_axis[start:end], grating_reg[start:end], **style.lsStimulusReg)
    axs_scoring['example_2'].plot(ca_time_axis[start:end]-1, vr_reg[start:end], **style.lsVrReg)

    start = int(660 * ca_fr)
    end = int(810 * ca_fr)
    scale_factor = 1
    loom_regs = stimulus_regs['bright_loom'] + stimulus_regs['dark_loom']
    axs_scoring['example_3'].plot(ca_time_axis[start:end], ca_traces[rois[-1]][start:end] * scale_factor, **style.lsCaTrace)
    axs_scoring['example_3'].plot(ca_time_axis[start:end], loom_regs[start:end] * scale_factor, **style.lsStimulusReg)
    axs_scoring['example_3'].plot(ca_time_axis[start:end]-1, vr_reg[start:end] * scale_factor, **style.lsVrReg)

    # Axes
    # X Limit
    x_lim_min = 60  # secs
    x_lim_max = 960
    for ax in axs:
        axs[ax].set_xlim(x_lim_min, x_lim_max)
        # Remove Axes
        axs[ax].axis('off')

    for ax in axs_scoring:
        axs_scoring[ax].axis('off')
        axs_scoring[ax].set_ylim(-1, 3)

    # Y Limit
    axs['ventral_root'].set_ylim(-12, 20)

    # Add Scale Bars
    # Time Scale Bar
    add_scale_bar(
        axs['ventral_root'], size=60, label='60 s', location=(0, 0.03), orientation='horizontal', color='black',
        linewidth=3, fontsize=10, padding=0.08)

    add_scale_bar(
        axs_scoring['example_1'], size=30, label='30 s', location=(0, 0.1), orientation='horizontal', color='black',
        linewidth=3, fontsize=10, padding=0.04)

    # Delta F over F Scale Bar
    add_scale_bar(
        axs['ca_trace_0'], size=2, label='2 df/f', location=(0.90, 0.50), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    add_scale_bar(
        axs['ventral_root_reg'], size=0.5, label='0.5 df/f', location=(0.90, 0.40), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    add_scale_bar(
        axs_scoring['example_3'], size=0.5, label='0.5 df/f', location=(0.80, 0.50), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    # Ventral Root mV Scale Bar
    add_scale_bar(
        axs['ventral_root'], size=5, label='5 mV', location=(0.95, 0.7), orientation='vertical', color='black',
        linewidth=3, fontsize=10, padding=0.02)

    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/example_traces3.pdf', dpi=600)
    plt.savefig('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging/figures/example_traces3.jpg', dpi=600)
    plt.close(fig)
    # plt.show()
    # exit()


def create_data_for_plotting():
    # Select Sweep Name
    sw = 'sw_31'

    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_events_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_events.h5'
    vr_dir = f'{base_dir}/ventral_root/selected_sweeps_aligned_vr_recordings.h5'
    vr_binaries_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries.csv'
    ca_labels_file = f'{base_dir}/data/df_f_data_labels.csv'

    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'
    ca_labels = pd.read_csv(ca_labels_file)

    # stimulus_protocols_dir = f'{base_dir}/protocols'
    stimulus_trace_dir = f'{base_dir}/stimulus_traces/stimulus_traces.csv'
    stimulus_onsets_dir = f'{base_dir}/stimulus_traces/{sw}_stimulus_onsets.csv'

    sw_stimulus_binaries = pd.read_csv(f'{Config.BASE_DIR}/stimulus_traces/{sw}_stimulus_binaries.csv', index_col=0)

    # Load Data
    vr_recordings = pd.read_hdf(vr_dir, key='data')
    vr_sampling_rate = 10000

    ca_df_f = pd.read_csv(ca_df_f_file)
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()

    # Detected VR Events for selected data set as a binary trace
    vr_binaries = pd.read_csv(vr_binaries_dir)
    vr_events = load_hdf5_as_dict(vr_events_dir)

    # Load Stimulus
    stimulus_traces = pd.read_csv(stimulus_trace_dir)

    rois_per_sweep = get_rois_per_sweep(ca_labels)

    # Sweep Name
    sw_rois = rois_per_sweep[sw].astype('str')

    plotting_data = dict()
    plotting_data['stimulus_trace'] = stimulus_traces[sw]
    plotting_data['stimulus_onsets'] = pd.read_csv(stimulus_onsets_dir, index_col=0)
    plotting_data['vr_recording'] = vr_recordings[sw]
    plotting_data['vr_events'] = vr_events[sw]
    plotting_data['vr_binary'] = vr_binaries[sw]
    plotting_data['stimulus_binary'] = sw_stimulus_binaries
    plotting_data['ca_traces'] = ca_df_f[sw_rois]
    plotting_data['vr_sampling_rate'] = pd.DataFrame([vr_sampling_rate])
    plotting_data['ca_sampling_rate'] = pd.DataFrame([ca_sampling_rate])

    # Store Plotting Data to HDD
    save_dict_as_hdf5(f'{base_dir}/figures/data/figure_example_traces.h5', plotting_data)
    print('=== PLOTTING DATA FOR FIGURE EXAMPLE TRACES STORED TO HDDD ====')


def main():
    # Create data for plotting and store it to HDD
    # create_data_for_plotting()
    # exit()

    # Load data
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    data_dir = f'{base_dir}/figures/data/figure_example_traces.h5'
    data = load_hdf5_as_dict(data_dir)
    std_z_stack_ref = plt.imread(f'{base_dir}/figures/data/STD_sw_02.png')
    # plot_example_traces(data, ref_img=std_z_stack_ref, rois=['31', '32', '37', '40', '41', '42'])
    # plot_example_traces(data, ref_img=std_z_stack_ref, rois=['456', '457', '459', '460', '461', '462'])
    plot_example_traces2(data, rois=['457', '456', '459', '460', '461'])


if __name__ == '__main__':
    main()
