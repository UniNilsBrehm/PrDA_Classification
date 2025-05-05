import os
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from utils import validate_directory, load_ca_data_headers_only

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'get_sweep_meta_data.py'

COLLECT META DATA FOR ALL SWEEPS
Collect all meta data information from the Ca Imaging recoding data, ventral root recordings and stimulus log files

Result: meta data csv file
index name fish z img_rec_start_date sampling_rate frame_dt VRR_start_date VRR_delay
0	sw_01	1	0	28:45.6	2.306230512	0.433608	28:45.6	-0.0337
1	sw_02	1	5	49:36.6	2.306230512	0.433608	49:36.7	0.0132

...

Requirements:
- directory called "time_stamps" containing the timestamp text file for each Ca Imaging Recoding (sweep)
- directory called "stimulation" containing all the stimulus log text files
- directory called "vrr_sweeps" containing all the ventral root recording  text files for each sweep
- directory called "data" containing the raw data for the Ca Imaging (raw_data.csv)
- directory called "meta_data" for storing the meta data csv file later on

Takes ca. 15 seconds to execute

Nils Brehm  -  2025
"""


def extract_ventral_root_info(vrr_start, date_str, rec_start_time_stamp):
    str_format = "%Y-%m-%d %H%M%S.%f"
    # Get Ventral Root Recording start
    # Load first vrr text file (01) and extract first timestamp
    vrr_start_timestamp = datetime.strptime(date_str + ' ' + str(vrr_start), str_format)

    # Calculate the delay between the Ca Imaging start and the VRR start for later alignment
    # If this is neg., vrr started earlier than imaging
    imaging_vr_diff = (vrr_start_timestamp - rec_start_time_stamp).total_seconds()
    return vrr_start_timestamp, imaging_vr_diff


def combine_dates_to_get_startpoint(date_str, sw_rec_onset):
    str_format = "%Y-%m-%d %H%M%S.%f"
    # Create Start of Recoding timestamp by combining info from ca timestamp file and stimulus log file
    return datetime.strptime(date_str + ' ' + str(sw_rec_onset), str_format)


def extract_meta_data_for_one_sweep(sw_info, date_str):
    # Get the Timestamp that indicates when the Ca Recording started (time of the day like 10:28:45.643)
    # e.g.: 102845.643 (hhmmss.ms)
    sw_rec_onset = sw_info[0].item()

    # Convert micro seconds into seconds for the frame interval
    sw_rec_dt = sw_info[1].item() / 1000 / 1000

    # Compute Frame Rate (Sampling Rate) by inverse of frame interval
    sw_rec_fr = 1 / sw_rec_dt

    # Load any stimulus log text file to get the day, month and year of the recording date
    # (Ca Timestamp only shows hour:minute:second.millisecond)
    str_format = "%Y-%m-%d %H%M%S.%f"

    # Create Start of Recoding timestamp by combining info from ca timestamp file and stimulus log file
    rec_start_time_stamp = datetime.strptime(date_str + ' ' + str(sw_rec_onset), str_format)
    return sw_rec_onset, sw_rec_dt, sw_rec_fr, rec_start_time_stamp


def create_meta_data_file(meta_data, meta_data_dir):
    # Combine everything in one metadata file
    meta_data_file = pd.DataFrame(
        meta_data,
        columns=['name', 'fish', 'z', 'img_rec_start_date', 'sampling_rate', 'frame_dt', 'VRR_start_date', 'VRR_delay']
    )

    # Store meta data as a csv file to disk
    meta_data_file.to_csv(f'{meta_data_dir}/meta_data.csv')

    # Store sampling rate (one for all recordings)
    fr = meta_data_file['sampling_rate'].unique()
    if len(fr) > 1:
        print('\n WARNING: FOUND MORE THAN ONE SAMPLING RATE!')
    else:
        fr_df = pd.DataFrame([fr[0], 1/fr[0]]).transpose()
        fr_df.columns = ['sampling_rate', 'frame_dt']
        fr_df.to_csv(f'{meta_data_dir}/sampling_rate.csv', index=False)
    # check = pickle_stuff(f'{meta_data_dir}/meta_data.pickle', meta_data_file)
    print(f'++++ STORED META DATA TO HDD ({meta_data_dir}) ++++')


def get_meta_data(recording_time_stamps_dir, ca_data_dir, stimulus_dir, vr_files_dir, meta_data_dir):
    # Validate Directories
    validate_directory(recording_time_stamps_dir)
    validate_directory(ca_data_dir)
    validate_directory(stimulus_dir)
    validate_directory(vr_files_dir)
    validate_directory(meta_data_dir)

    # Get a List of all timestamps files of all sweeps
    list_of_recording_time_stamps = os.listdir(recording_time_stamps_dir)

    # Load raw data Ca data csv file (each col is a roi data trace), Rows 0 to 4 contain meta data
    # Each Col is one ROI
    # Row 0: Fish Number
    # Row 1: Sweep Number
    # Row 2: ROI Number
    # Row 3: Y-X Position
    # Row 4: Z Position (Plane Number)
    # Row 5-N: Fluorescence Values
    # ca_data_labels = pd.read_csv(ca_data_dir, header=None, index_col=False, nrows=5)
    ca_data_labels = load_ca_data_headers_only(ca_data_dir)

    meta_data = list()
    # Loop over all available sweeps and get the stimulus files
    for f in list_of_recording_time_stamps:
        # Get the sweep name (sw_01) from the text file names
        sw = f[:5]

        # Load the timestamp text file
        # COl 0: timestamp when the Ca Recording started
        # COL 1: frame interval in micro seconds
        sw_info = pd.read_csv(f'{recording_time_stamps_dir}/{f}', sep='\t', header=None)

        # Load any stimulus log text file to get the day, month and year of the recording date
        # (Ca Timestamp only shows hour:minute:second.millisecond)
        date_str = pd.read_csv(f'{stimulus_dir}/{sw}/sw_flash.txt', sep='\t', header=None).iloc[0, 1][:10]

        sw_rec_onset, sw_rec_dt, sw_rec_fr, rec_start_time_stamp = extract_meta_data_for_one_sweep(sw_info, date_str)

        rec_start_time_stamp = combine_dates_to_get_startpoint(date_str, sw_rec_onset)

        # Get Ventral Root Recording start
        # Load first vrr text file (01) and extract first timestamp
        vrr_start = pd.read_csv(f'{vr_files_dir}/{sw}_01_ephys_vrr.txt', sep='\t', header=None).iloc[0, 3]
        vrr_start_timestamp, imaging_vr_diff = extract_ventral_root_info(vrr_start, date_str, rec_start_time_stamp)

        # Get the Fish ID Number
        fish_nr = ca_data_labels.loc[0, ca_data_labels.loc[1] == sw].iloc[0]

        # Get the z-plane
        z_plane = ca_data_labels.loc[4, ca_data_labels.loc[1] == sw].iloc[0]

        # Combine everything into one metadata list containing all sweeps
        # [sweep, fish, recording start, frame rate, frame time, vrr start, vrr_delay]
        meta_data.append([sw, fish_nr, z_plane, str(rec_start_time_stamp), sw_rec_fr, sw_rec_dt, str(vrr_start_timestamp), imaging_vr_diff])

    # Combine everything in one metadata file and store it to HDD
    create_meta_data_file(meta_data, meta_data_dir)


def main():
    base_dir = Config.BASE_DIR
    print(f'\n ==== BASE DIR set to: {base_dir} ==== \n')
    get_meta_data(
        recording_time_stamps_dir=f'{base_dir}/time_stamps',
        ca_data_dir=f'{base_dir}/data/raw_data.csv',
        stimulus_dir=f'{base_dir}/stimulation',
        vr_files_dir=f'{base_dir}/vrr_sweeps',
        meta_data_dir=f'{base_dir}/meta_data'
                  )


if __name__ == '__main__':
    import timeit
    n = 1
    result = timeit.timeit(stmt='main()', globals=globals(), number=n)
    # calculate the execution time
    # get the average execution time
    print(f"Execution time is {result / n} seconds")