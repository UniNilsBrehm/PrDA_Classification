import os
import pandas as pd
import numpy as np
from IPython import embed
from utils import validate_directory, convert_time_stamps_to_secs, pickle_stuff
from joblib import Parallel, delayed
"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'ventral_root_pre_processing.py'

Convert the chunked ventral root recording files (.txt) into one file.
Due to the way the time stamps are stored in these files it takes some time to convert to seconds.
The Script will use parallel computing (multi-cores). Still it will take some time to finish!
Either all .txt files are in one directory or every sweep has its own directory containing the ventral root 
recording files (.txt). 
If a sweep is missing its first recording you can label the folder with "..._fm" to indicate this.
Due to the way the txt files are stored, there will be a variable gap of some seconds between each text file.
This information is lost, however we take care of it by filling the gaps with zeros.

Result: 
    - /ventral_root/all_vr_recordings.h5' (around 4 GB)

...

Requirements:
    - /ventral_root/
    - /vrr_sweeps/ (containing all ventral root text files from Olympus setup).
    
Nils Brehm  -  2025
"""


def transform_ventral_root_recording(vr_files, vr_fr, first_file_missing=0):
    # vr_files is a list of ventral root recording text files of one sweep
    vr_files = sorted(vr_files)

    # Get First Recording Text File
    f_name_01 = vr_files[0]

    # Load Text File
    first_vr_text_file = pd.read_csv(f_name_01, sep='\t', header=None)

    # Get the Voltage values
    vr_values = first_vr_text_file.iloc[:, 0].to_numpy()

    # Get end time point for finding the gap to the next recording file
    t_last = convert_time_stamps_to_secs(first_vr_text_file.iloc[-1, 3], method=0)

    # Loop over the rest of the vr recording text files
    for f_name in vr_files[1:]:
        # check file size
        if os.path.getsize(f_name) <= 10:
            print('')
            print('WARNING')
            print(f'{f_name}: File Size is too small. Something is wrong with this file. Please check!')
            print('Will skip this file and set all values to zero')
            print('')
            continue

        # Load ventral root text file
        vr_text_file = pd.read_csv(f_name, sep='\t', header=None)

        # Compute time distance between the end of the last vr recording and the start of this one (in seconds)
        t_rel_distance = convert_time_stamps_to_secs(vr_text_file.iloc[0, 3], method=0) - t_last

        # Fill the Gap between Recordings with zeros
        n_zeros = np.zeros(int(vr_fr * t_rel_distance))
        values = vr_text_file.iloc[:, 0].to_numpy()
        vr_values = np.append(vr_values, n_zeros)
        vr_values = np.append(vr_values, values)

        # store last time point of this recording for the next round
        t_last = convert_time_stamps_to_secs(vr_text_file.iloc[-1, 3], method=0)

    # If the first recording file is missing, correct for it now
    if first_file_missing > 0:
        # We know one recording is always 60 seconds long
        # Reset time so that it starts at 60 seconds since the first recording is missing
        n_zeros = np.zeros(int(vr_fr * first_file_missing))
        vr_values = np.append(n_zeros, vr_values)

    return vr_values


def transform_ventral_root_parallel(save_dir, base_dir, rec_dur, vr_fr, separate_dirs, sw):
    # Get file list with all the ventral root recoding text files
    f_path = []

    # If every sweep has its own folder containing the text files
    if separate_dirs:
        f_names = os.listdir(f'{base_dir}/{sw}')
        # f_names = list(np.sort(f_names))
        f_names = sorted(f_names)

        for _, n in enumerate(f_names):
            f_path.append(f'{base_dir}/{sw}/{n}')

    # If all text files of all sweeps are in one and the same directory
    else:
        all_files = os.listdir(base_dir)
        f_names = list(np.sort([i for i in all_files if i.startswith(sw)]))
        for _, n in enumerate(f_names):
            f_path.append(f'{base_dir}/{n}')

    # Check for missing files: Sweep that are missing the first recording must be labeled with the suffix: "fm"
    # We assume that every recording is 60 seconds long
    firs_rec_missing = 0
    if sw.endswith('fm'):
        sw = sw[:-3]
        firs_rec_missing = rec_dur  # secs
        print(f'First Recording missing in sweep {sw}')
        print(f'Will correct for that, assuming that each recording has a duration of 60 seconds!')

    print(f'START PROCESSING: {sw}')
    vr_trace = transform_ventral_root_recording(f_path, vr_fr=vr_fr, first_file_missing=firs_rec_missing)

    vr_trace_export = pd.DataFrame(columns=[sw])
    vr_trace_export[sw] = vr_trace

    if save_dir is not None:
        to_dir = f'{save_dir}/{sw}_ventral_root.csv'
        vr_trace_export.to_csv(to_dir, index=False)
        print(f'Ventral Root of Sweep: {sw} stored to HDD')
    # result = {sw: vr_trace}
    return vr_trace_export


def main():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    save_dir = f'{base_dir}/ventral_root'
    vr_files_dir = f'{base_dir}/vrr_sweeps'

    # SETTINGS
    separate_dirs = False  # True if every sweep is in its own folder
    vr_rec_dur = 60  # the duration of one recording text file in seconds
    vr_fr = 10000  # VR Sampling Rate in Hz

    print(f'Ventral Root Recording Files in: {vr_files_dir}')
    print(f'Store result to: {save_dir}')
    print('++++ START PROCESSING +++')
    print('This can take some time ...')
    print('This relies heavily on CPU, RAM and HDD. HDD is normally the bottleneck, so make sure to use a fast one!')
    print('... Please Wait ...')
    print('')

    # Check if necessary directories are there
    validate_directory(save_dir)
    validate_directory(vr_files_dir)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Select single sweeps (FOR DEBUGGING)
    # sw = 'sw_01'
    # r = transform_ventral_root_parallel(
    #     save_dir=None, base_dir=vr_files_dir, rec_dur=vr_rec_dur, vr_fr=vr_fr, separate_dirs=separate_dirs, sw=sw)
    # embed()
    # exit()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Process all sweeps in parallel
    if separate_dirs:
        sweep_numbers = os.listdir(vr_files_dir)
    else:
        sweep_numbers = np.unique([i[:5] for i in os.listdir(vr_files_dir)])

    # START PROCESSING IN PARALLEL
    results = Parallel(n_jobs=-2)(delayed(
        transform_ventral_root_parallel)(None, vr_files_dir, vr_rec_dur, vr_fr, separate_dirs, i) for i in sweep_numbers)

    print('STORING DATA TO HDD')

    # Prepare Data to be stored on HDD
    vr_recordings = pd.concat(results, axis=1)

    # Replace NaN with 0
    vr_recordings.fillna(0, inplace=True)

    # # Store csv file to HDD
    # vr_recordings.to_csv(f'{temp_data_dir}/all_vr_recordings.csv', index=False)

    # Store hdf5 file to HDD
    vr_recordings.to_hdf(f'{save_dir}/all_vr_recordings.h5', key='data', mode='w')

    # Load from HDF5 file
    # loaded_df = pd.read_hdf('data.h5', key='data')

    # vr_recordings = dict()
    # for res in results:
    #     sw = str(list(res.keys())[0])
    #     vr_recordings[sw] = res[sw]
    #     a = res[sw]
    #     a.column = sw
    #     vr_recordings.column = sw
    #
    #     # vr_recordings[sw] = df1.read_csv(f'{vr_dir}/{f_name}')
    #
    # # Store data to HDD
    # pickle_stuff(vr_recs_pickle_file, data=vr_recordings)

    print('++++ FINISHED PROCESSING ++++')


if __name__ == '__main__':
    main()
