import numpy as np
import pandas as pd
from utils import load_ca_meta_data, align_traces_via_padding
from IPython import embed

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'ventral_root_align_recordings_with_ca_imaging.py'

Align in time the ventral root recordings with the ca imaging data for each sweep. Store time aligned ventral root
recordings to disk for further analysis.


Result: 
    - /ventral_root/aligned_vr_recordings.h5' (around 4 GB)
    - /ventral_root/selected_sweeps_aligned_vr_recordings.h5'

...

Requirements:
    - /ventral_root/all_vr_recordings.h5
    - /PrDA_good_sweeps.csv'

Nils Brehm  -  2025
"""


def aligning_vr_traces(meta_data, vr_data, vr_sampling_rate):
    # Loop over sweeps and add the time delay to the time axis to align it to the imaging recording
    all_aligned_vr = pd.DataFrame()
    for index, row in meta_data.iterrows():
        sw = row['name']
        vr_delay = row['VRR_delay']
        vr = vr_data[sw]
        aligned_vr = align_traces_via_padding(vr, vr_delay, vr_sampling_rate)
        all_aligned_vr[sw] = aligned_vr
        # import matplotlib.pyplot as plt
        # time_axis = np.linspace(0, vr_data.shape[0] * vr_sampling_rate, vr_data.shape[0])
        # plt.plot(time_axis, vr, 'b')
        # plt.plot(time_axis, aligned_vr, 'r--')
        # plt.show()

    return all_aligned_vr


def main():
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_dir = f'{base_dir}/ventral_root/all_vr_recordings.h5'
    meta_data_file = f'{base_dir}/meta_data/meta_data.csv'

    # Load VR data and metadata
    vr_data = pd.read_hdf(vr_dir, key='data')
    meta_data = load_ca_meta_data(meta_data_file)
    vr_fr = 10000  # VR Sampling Rate in Hz
    vr_aligned = aligning_vr_traces(meta_data, vr_data, vr_fr)

    # Store aligned VR to HDD
    print('')
    print('++++ STORING ALIGNED VENTRAL ROOT RECORDINGS TO HDD ++++')
    print('This can take some minutes ...')
    print('')
    vr_aligned.to_hdf(f'{base_dir}/ventral_root/aligned_vr_recordings.h5', key='data', mode='w')

    # Store the selected sweeps (good recordings) separately
    print('')
    print('++++ STORING A SELECTION OF ALIGNED VENTRAL ROOT RECORDINGS TO HDD SEPARATELY ++++')
    print('This can take some minutes ...')
    print('')
    selected_sweeps = pd.read_csv(f'{base_dir}/PrDA_good_sweeps.csv')['sweep_name']
    vr_aligned_selected = vr_aligned[selected_sweeps]
    vr_aligned_selected.to_hdf(f'{base_dir}/ventral_root/selected_sweeps_aligned_vr_recordings.h5', key='data', mode='w')


if __name__ == '__main__':
    import timeit
    n = 1
    result = timeit.timeit(stmt='main()', globals=globals(), number=n)
    # calculate the execution time
    # get the average execution time
    print(f"Execution time is {(result/60) / n: .2f} minutes")
