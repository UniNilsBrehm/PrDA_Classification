from utils import load_ca_meta_data
import os
import pandas as pd
import numpy as np
from IPython import embed
from config import Config
from datetime import datetime, timedelta


"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'stimulus_pre_processing.py'

Prepare stimulus data for further analysis
Stimuli are:
- moving_target
- grating_appears
- grating_0
- grating_180
- grating_disappears
- bright loom
- dark loom
- bright flash
- dark flash

Result: 
- protocols/sw_stimulus_protocol.csv
- stimulus_traces/sw_stimulus_onsets.csv
- stimulus_traces/sw_stimulus_offsets.csv
- stimulus_traces/sw_stimulus_binaries.csv
- stimulus_traces/stimulus_traces.csv

...

Requirements:
    - /stimulation
    - /meta_data/meta_data.csv
    - /meta_data/sampling_rate.csv
    - /data/df_f_data.csv

Nils Brehm  -  2025
"""


def handle_time_stamps(datetime_string):
    # String looks like this:
    # 2023-07-03 11:13:48.078000
    if len(datetime_string) == 19:
        datetime_string = datetime_string + '.000000'
    str_format = "%Y-%m-%d %H:%M:%S.%f"
    date_object = datetime.strptime(datetime_string, str_format)
    return date_object


def process_flash(f, rec_start_time_stamp):
    # FLASH
    # Col 0: 0=dark, 1=bright flash, -1=dark flash,
    # Col 1: Time Stamp ("2024-03-04 10:42:24.56000")

    # Get the bright flash (Col 0 == 1) Time Stamp
    bright_flash_onset_str = f[f[0] == 1][1].item()
    bright_flash_onset = handle_time_stamps(bright_flash_onset_str)

    # Get the dark flash (Col 0 == -1) Time Stamp
    dark_flash_onset = handle_time_stamps(f[f[0] == -1][1].item())

    # Now get the onsets of the flash in seconds after recording start
    bright_flash_onset_seconds = (bright_flash_onset - rec_start_time_stamp).total_seconds()
    dark_flash_onset_seconds = (dark_flash_onset - rec_start_time_stamp).total_seconds()

    # Now create readable output for a protocol file
    flash = list()
    flash.append(['flash', 'bright', 'ON', bright_flash_onset_seconds, str(bright_flash_onset)])
    flash.append(['flash', 'dark', 'ON', dark_flash_onset_seconds, str(dark_flash_onset)])

    onsets = [bright_flash_onset_seconds, dark_flash_onset_seconds]
    offsets = [bright_flash_onset_seconds + 1, dark_flash_onset_seconds + 1]

    return flash, onsets, offsets


def process_grating(f, rec_start_time_stamp):
    # Gratings
    # Col 1: Orientation in degrees
    # Col 2: Time Stamp ("2024-03-04 10:42:24.56000")
    # Col 3: Temporal Frequency
    # Col 4: Spatial Frequency

    # Get Time Stamps (this is hard coded in the stimulus file)
    appears = handle_time_stamps(f[2][0])
    grating_0_starts_moving = handle_time_stamps(f[2][1])
    grating_0_stops_moving = handle_time_stamps(f[2][2])
    grating_180_starts_moving = handle_time_stamps(f[2][5])
    grating_180_stops_moving = handle_time_stamps(f[2][6])
    disappears = handle_time_stamps(f[2][7])

    # Now get the onsets in seconds after recording start
    appears_sec = (appears - rec_start_time_stamp).total_seconds()
    grating_0_starts_moving_sec = (grating_0_starts_moving - rec_start_time_stamp).total_seconds()
    grating_0_stops_moving_sec = (grating_0_stops_moving - rec_start_time_stamp).total_seconds()
    grating_180_starts_moving_sec = (grating_180_starts_moving - rec_start_time_stamp).total_seconds()
    grating_180_stops_moving_sec = (grating_180_stops_moving - rec_start_time_stamp).total_seconds()
    disappears_sec = (disappears - rec_start_time_stamp).total_seconds()

    # Now create readable output for a protocol file
    gratings = list()
    gratings.append(['grating', '0', 'appears',  appears_sec, str(appears)])
    gratings.append(['grating', '0', 'start', grating_0_starts_moving_sec, str(grating_0_starts_moving)])
    gratings.append(['grating', '0', 'stop', grating_0_stops_moving_sec, str(grating_0_stops_moving)])
    gratings.append(['grating', '180', 'start', grating_180_starts_moving_sec, str(grating_180_starts_moving)])
    gratings.append(['grating', '180', 'stop', grating_180_stops_moving_sec, str(grating_180_stops_moving)])
    gratings.append(['grating', '180', 'disappears',  disappears_sec, str(disappears)])

    onsets = [appears_sec, grating_0_starts_moving_sec, grating_180_starts_moving_sec, disappears_sec]
    offsets = [appears_sec + 1, grating_0_stops_moving_sec, grating_180_stops_moving_sec, disappears_sec + 1]

    return gratings, onsets, offsets


def process_moving_target(f, rec_start_time_stamp):
    # Moving Target (small): 2 repetitions with 60 s interval
    # Col 0: dot position
    # Col 1: ignore
    # Col 2: ignore
    # Col 3: Time Stamp ("2024-03-04 10:42:24.56000")

    # Convert Time Stamps to seconds after recording start
    time_axis = []
    for ts in f[3]:
        time_axis.append((handle_time_stamps(ts) - rec_start_time_stamp).total_seconds())
    # Compute difference in time to get the interval between the two repetitions
    time_diff = np.diff(time_axis)
    idx_interval = np.where(time_diff > 50)[0][0]

    # Find onsets and offsets
    mt_01_start_timestamp = f[3].iloc[0]
    mt_01_stop_timestamp = f[3].iloc[idx_interval]
    mt_02_start_timestamp = f[3].iloc[idx_interval+1]
    mt_02_stop_timestamp = f[3].iloc[-1]

    mt_01_start_secs = (handle_time_stamps(mt_01_start_timestamp) - rec_start_time_stamp).total_seconds()
    mt_01_stop_secs = (handle_time_stamps(mt_01_stop_timestamp) - rec_start_time_stamp).total_seconds()
    mt_02_start_secs = (handle_time_stamps(mt_02_start_timestamp) - rec_start_time_stamp).total_seconds()
    mt_02_stop_secs = (handle_time_stamps(mt_02_stop_timestamp) - rec_start_time_stamp).total_seconds()

    moving_target = list()
    moving_target.append(['moving_target', '01', 'start',  mt_01_start_secs, str(mt_01_start_timestamp)])
    moving_target.append(['moving_target', '01', 'stop',  mt_01_stop_secs, str(mt_01_stop_timestamp)])
    moving_target.append(['moving_target', '02', 'start',  mt_02_start_secs, str(mt_02_start_timestamp)])
    moving_target.append(['moving_target', '02', 'stop',  mt_02_stop_secs, str(mt_02_stop_timestamp)])

    onsets = [mt_01_start_secs, mt_02_start_secs]
    offsets = [mt_01_stop_secs, mt_02_stop_secs]

    return moving_target, onsets, offsets


def process_dark_looming(f, rec_start_time_stamp):
    # Dark Looming (dark dot on bright screen)
    # Col 0: Dot Size
    # Col 1: Time Stamp ("2024-03-04 10:42:24.56000")

    # Convert Time Stamps to seconds after recording start
    time_axis = []
    for ts in f[1]:
        time_axis.append((handle_time_stamps(ts) - rec_start_time_stamp).total_seconds())

    # This is hard coded in the stimulus file
    looming_start_timestamp = f[1].iloc[0]
    looming_stop_timestamp = f[1].iloc[-3]
    looming_static_screen_timestamp = f[1].iloc[-2]  # when the static screen (white screen) stops

    looming_start_sec = (handle_time_stamps(looming_start_timestamp) - rec_start_time_stamp).total_seconds()
    looming_stop_sec = (handle_time_stamps(looming_stop_timestamp) - rec_start_time_stamp).total_seconds()
    looming_static_screen_sec = (handle_time_stamps(looming_static_screen_timestamp) - rec_start_time_stamp).total_seconds()

    dark_looming = list()
    dark_looming.append(['loom', 'dark', 'start',  looming_start_sec, str(looming_start_timestamp)])
    dark_looming.append(['loom', 'dark', 'stop',  looming_stop_sec, str(looming_stop_timestamp)])
    dark_looming.append(['loom', 'dark', 'static_end',  looming_static_screen_sec, str(looming_static_screen_timestamp)])

    onsets = looming_start_sec
    offsets = looming_stop_sec

    return dark_looming, onsets, offsets


def process_bright_looming(f, rec_start_time_stamp):
    # Bright Looming (white dot on dark background)
    # Col 0: Dot Size
    # Col 1: Time Stamp ("2024-03-04 10:42:24.56000")

    # Convert Time Stamps to seconds after recording start
    time_axis = []
    for ts in f[1]:
        time_axis.append((handle_time_stamps(ts) - rec_start_time_stamp).total_seconds())

    # This is hard coded in the stimulus file
    looming_start_timestamp = f[1].iloc[0]
    looming_stop_timestamp = f[1].iloc[-3]
    looming_static_screen_timestamp = f[1].iloc[-2]  # when the static screen (dark screen) stops

    looming_start_sec = (handle_time_stamps(looming_start_timestamp) - rec_start_time_stamp).total_seconds()
    looming_stop_sec = (handle_time_stamps(looming_stop_timestamp) - rec_start_time_stamp).total_seconds()
    looming_static_screen_sec = (handle_time_stamps(looming_static_screen_timestamp) - rec_start_time_stamp).total_seconds()

    bright_looming = list()
    bright_looming.append(['loom', 'bright', 'start',  looming_start_sec, str(looming_start_timestamp)])
    bright_looming.append(['loom', 'bright', 'stop',  looming_stop_sec, str(looming_stop_timestamp)])
    bright_looming.append(['loom', 'bright', 'static_end',  looming_static_screen_sec, str(looming_static_screen_timestamp)])

    onsets = looming_start_sec
    offsets = looming_stop_sec

    return bright_looming, onsets, offsets


def reconstruct_bright_loom(dark_loom_starts, rec_start_time_stamp):
    # Difference to ground truth was 10-20 ms
    # Loom static screen stops when dark loom starts (delay of ca 100 ms)
    looming_static_screen_timestamp = handle_time_stamps(dark_loom_starts) - timedelta(milliseconds=100)

    # Loom starts 60 seconds before static ends
    looming_stop_timestamp = looming_static_screen_timestamp - timedelta(seconds=60)

    # Loom duration is 2 seconds
    looming_start_timestamp = looming_stop_timestamp - timedelta(seconds=2)

    # In seconds
    looming_start_sec = (looming_start_timestamp - rec_start_time_stamp).total_seconds()
    looming_stop_sec = (looming_stop_timestamp - rec_start_time_stamp).total_seconds()
    looming_static_screen_sec = (looming_static_screen_timestamp - rec_start_time_stamp).total_seconds()

    bright_looming = list()
    bright_looming.append(['loom', 'bright', 'start',  looming_start_sec, str(looming_start_timestamp)])
    bright_looming.append(['loom', 'bright', 'stop',  looming_stop_sec, str(looming_stop_timestamp)])
    bright_looming.append(['loom', 'bright', 'static_end',  looming_static_screen_sec, str(looming_static_screen_timestamp)])

    onsets = looming_start_sec
    offsets = looming_stop_sec

    return bright_looming, onsets, offsets


def create_stimulus_protocol(meta_data, sw, sw_dir):
    list_of_stimulus_files = os.listdir(sw_dir)
    # Get Imaging Recording Onset Time
    sw_rec_onset = handle_time_stamps(meta_data[meta_data['name'] == sw]['img_rec_start_date'].item())

    all_stimuli = list()
    stimulus_onsets = pd.DataFrame()
    stimulus_offsets = pd.DataFrame()

    # Loop over stimulus files and extract information to create a protocol
    for s_file_name in list_of_stimulus_files:
        s_type = s_file_name[3:-4]  # remove sw and .txt
        # Open stimulus file
        s_file = pd.read_csv(f'{sw_dir}/{s_file_name}', sep='\t', header=None)
        if s_type == 'flash':
            # bright_flash, dark_flash
            flash, onsets, offsets = process_flash(s_file, sw_rec_onset)
            all_stimuli.extend(flash)
            stimulus_onsets['bright_flash'] = [onsets[0]]
            stimulus_onsets['dark_flash'] = [onsets[1]]

            stimulus_offsets['bright_flash'] = [offsets[0]]
            stimulus_offsets['dark_flash'] = [offsets[1]]

        elif s_type == 'grating':
            # stimulus_onsets:
            # [appears_sec, grating_0_starts_moving_sec, grating_180_starts_moving_sec, disappears_sec]
            gratings, onsets, offsets = process_grating(s_file, sw_rec_onset)
            all_stimuli.extend(gratings)
            stimulus_onsets['grating_appears'] = [onsets[0]]
            stimulus_onsets['grating_0'] = [onsets[1]]
            stimulus_onsets['grating_180'] = [onsets[2]]
            stimulus_onsets['grating_disappears'] = [onsets[3]]

            stimulus_offsets['grating_appears'] = [offsets[0]]
            stimulus_offsets['grating_0'] = [offsets[1]]
            stimulus_offsets['grating_180'] = [offsets[2]]
            stimulus_offsets['grating_disappears'] = [offsets[3]]

        elif s_type == 'movingtargetsmall':
            moving_target, onsets, offsets = process_moving_target(s_file, sw_rec_onset)
            all_stimuli.extend(moving_target)
            stimulus_onsets['moving_target_01'] = [onsets[0]]
            stimulus_onsets['moving_target_02'] = [onsets[1]]

            stimulus_offsets['moving_target_01'] = [offsets[0]]
            stimulus_offsets['moving_target_02'] = [offsets[1]]

        elif s_type == 'looming':
            dark_looming, onsets, offsets = process_dark_looming(s_file, sw_rec_onset)
            all_stimuli.extend(dark_looming)
            stimulus_onsets['dark_loom'] = [onsets]
            stimulus_offsets['dark_loom'] = [offsets]

        elif s_type == 'looming_rev':  # Not all sweeps have this due to some logging errors
            bright_looming, onsets, offsets = process_bright_looming(s_file, sw_rec_onset)
            all_stimuli.extend(bright_looming)
            stimulus_onsets['bright_loom'] = [onsets]
            stimulus_offsets['bright_loom'] = [offsets]

        else:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'WARNING: Did not recognize this stimulus file: {s_file_name}')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # If there is no "looming_rec" due to some logging errors
    if 'sw_looming_rev.txt' not in list_of_stimulus_files:
        print(f'{sw} - COULD NOT FIND: "sw_looming_rev.txt", will interpolate this stimulus')
        bright_looming, onsets, offsets = reconstruct_bright_loom(dark_looming[0][4], sw_rec_onset)
        all_stimuli.extend(bright_looming)
        stimulus_onsets['bright_loom'] = [onsets]
        stimulus_offsets['bright_loom'] = [offsets]

    # Get recording start time stamp (since this is stored without year and month we have to add this from stimulus)
    all_stimuli.append([sw, 'recording', 'start', 0, str(sw_rec_onset)])

    # Now combine everything into one protocol file
    stimulus_protocol = pd.DataFrame(all_stimuli)
    stimulus_protocol.sort_values(by=3, inplace=True)
    stimulus_onsets = stimulus_onsets.reindex(columns=stimulus_onsets.iloc[0].sort_values().index)
    stimulus_offsets = stimulus_offsets.reindex(columns=stimulus_offsets.iloc[0].sort_values().index)

    return stimulus_protocol, stimulus_onsets, stimulus_offsets


def create_stimulus_traces(stimulus_protocol, ca_time_axis, ca_sampling_rate, sw):
    # Create a stimulus trace for plotting etc.
    stimulus_trace = np.zeros_like(ca_time_axis)
    # stimulus_binaries = dict()
    stimulus_binaries = pd.DataFrame()
    # Note: all the used indices below are hard coded in the protocol file

    # 1. Moving Target
    idx_start = np.ceil(stimulus_protocol.iloc[1, 3] * ca_sampling_rate).astype('int')
    idx_end = np.ceil(stimulus_protocol.iloc[2, 3] * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    moving_target_binary = np.zeros_like(ca_time_axis)
    moving_target_binary[idx_start:idx_end] = 1
    stimulus_binaries['moving_target_01'] = moving_target_binary

    # 2. Moving Target
    idx_start = np.ceil(stimulus_protocol.iloc[3, 3] * ca_sampling_rate).astype('int')
    idx_end = np.ceil(stimulus_protocol.iloc[4, 3] * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    moving_target_binary = np.zeros_like(ca_time_axis)
    moving_target_binary[idx_start:idx_end] = 1
    stimulus_binaries['moving_target_02'] = moving_target_binary

    # Grating 0: Appears
    prot_gratings = stimulus_protocol[stimulus_protocol[0] == 'grating']
    prot = prot_gratings[prot_gratings[1] == '0']
    idx_start = np.ceil(prot[prot[2] == 'appears'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_start + 1] = 1
    grating_appears_binary = np.zeros_like(ca_time_axis)
    grating_appears_binary[idx_start:idx_start + 1] = 1
    stimulus_binaries['grating_appears'] = grating_appears_binary

    # Grating 0: Moves
    idx_start = np.ceil(prot[prot[2] == 'start'][3].item() * ca_sampling_rate).astype('int')
    idx_end = np.ceil(prot[prot[2] == 'stop'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    grating_0_binary = np.zeros_like(ca_time_axis)
    grating_0_binary[idx_start:idx_end] = 1
    stimulus_binaries['grating_0'] = grating_0_binary

    # Grating 180: Moves
    prot = prot_gratings[prot_gratings[1] == '180']
    idx_start = np.ceil(prot[prot[2] == 'start'][3].item() * ca_sampling_rate).astype('int')
    idx_end = np.ceil(prot[prot[2] == 'stop'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    grating_180_binary = np.zeros_like(ca_time_axis)
    grating_180_binary[idx_start:idx_end] = 1
    stimulus_binaries['grating_180'] = grating_180_binary

    # Grating 0: Disappears
    idx_start = np.ceil(prot[prot[2] == 'disappears'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_start + 1] = 1
    grating_disappears_binary = np.zeros_like(ca_time_axis)
    grating_disappears_binary[idx_start:idx_start + 1] = 1
    stimulus_binaries['grating_disappears'] = grating_disappears_binary

    # Bright Loom (dot expands)
    prot_looms = stimulus_protocol[stimulus_protocol[0] == 'loom']
    prot = prot_looms[prot_looms[1] == 'bright']
    idx_start = np.ceil(prot[prot[2] == 'start'][3].item() * ca_sampling_rate).astype('int')
    idx_end = np.ceil(prot[prot[2] == 'stop'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    bright_loom_binary = np.zeros_like(ca_time_axis)
    bright_loom_binary[idx_start:idx_end] = 1
    stimulus_binaries['bright_loom'] = bright_loom_binary

    # Bright Loom static screen (Bright Screen)
    # idx_start = idx_end
    # idx_end = np.ceil(prot[prot[2] == 'static_end'][3].item() * ca_sampling_rate).astype('int')
    # stimulus_trace[idx_start:idx_end] = 0.75

    # Dark Loom (dot expands)
    prot = prot_looms[prot_looms[1] == 'dark']
    idx_start = np.ceil(prot[prot[2] == 'start'][3].item() * ca_sampling_rate).astype('int')
    idx_end = np.ceil(prot[prot[2] == 'stop'][3].item() * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    dark_loom_binary = np.zeros_like(ca_time_axis)
    dark_loom_binary[idx_start:idx_end] = 1
    stimulus_binaries['dark_loom'] = dark_loom_binary

    # Dark Loom static screen (Dark Screen)
    # idx_start = idx_end
    # idx_end = np.ceil(prot[prot[2] == 'static_end'][3].item() * ca_sampling_rate).astype('int')
    # stimulus_trace[idx_start:idx_end] = 0.25

    # Bright Flash
    idx_start = np.ceil(stimulus_protocol.iloc[-2, 3] * ca_sampling_rate).astype('int')
    idx_end = idx_start + 1
    # idx_end = np.ceil(stimulus_protocol.iloc[-1, 3] * ca_sampling_rate).astype('int')
    stimulus_trace[idx_start:idx_end] = 1
    bright_flash_binary = np.zeros_like(ca_time_axis)
    bright_flash_binary[idx_start:idx_end] = 1
    stimulus_binaries['bright_flash'] = bright_flash_binary

    # Dark Flash
    idx_start = np.ceil(stimulus_protocol.iloc[-1, 3] * ca_sampling_rate).astype('int')
    # stimulus_trace[idx_start:] = 0.25
    idx_end = idx_start + 1
    stimulus_trace[idx_start:idx_end] = 1
    dark_flash_binary = np.zeros_like(ca_time_axis)
    dark_flash_binary[idx_start:idx_end] = 1
    stimulus_binaries['dark_flash'] = dark_flash_binary

    # Store Stimulus Trace to HDD
    stimulus_trace_file = pd.DataFrame()
    stimulus_trace_file[sw] = stimulus_trace

    return stimulus_trace_file, stimulus_binaries


def main():
    # Directories
    base_dir = Config.BASE_DIR
    stimulus_dir = f'{base_dir}/stimulation'
    meta_data_file = f'{base_dir}/meta_data/meta_data.csv'
    meta_data = load_ca_meta_data(meta_data_file)
    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'
    ca_df_f = pd.read_csv(ca_df_f_file)

    ca_time_axis = np.linspace(0, ca_df_f.shape[0] / ca_sampling_rate, ca_df_f.shape[0])

    # Get a list of all sweeps
    sweeps = os.listdir(stimulus_dir)

    # stimulus_binaries_dict = dict()
    stimulus_traces_df = pd.DataFrame()

    # Loop over all available sweeps and get the stimulus files
    for sw in sweeps:
        sw_dir = f'{stimulus_dir}/{sw}'
        stimulus_protocol, stimulus_onsets, stimulus_offsets = create_stimulus_protocol(meta_data, sw, sw_dir)
        # Store Stimulus Protocol for this sweep to HDD
        stimulus_protocol.to_csv(f'{base_dir}/protocols/{sw}_stimulus_protocol.csv', header=False)
        stimulus_onsets.to_csv(f'{base_dir}/stimulus_traces/{sw}_stimulus_onsets.csv')
        stimulus_offsets.to_csv(f'{base_dir}/stimulus_traces/{sw}_stimulus_offsets.csv')

        stimulus_trace_file, stimulus_binaries = create_stimulus_traces(
            stimulus_protocol, ca_time_axis, ca_sampling_rate, sw
        )

        stimulus_binaries.to_csv(f'{base_dir}/stimulus_traces/{sw}_stimulus_binaries.csv')
        stimulus_traces_df[sw] = stimulus_trace_file
        print(f'{sw}: STIMULUS PROTOCOL AND BINARY TRACE STORED TO HDD')
        print('')

    # Store Results to HDD
    stimulus_traces_df.to_csv(f'{base_dir}/stimulus_traces/stimulus_traces.csv')
    print('')
    print('++++ FINISHED STIMULUS PREPROCESSING ++++')


if __name__ == '__main__':
    main()
