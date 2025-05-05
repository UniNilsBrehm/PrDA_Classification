import pandas as pd
import numpy as np
from utils import load_raw_ca_data_only, load_ca_data_headers_only, load_ca_meta_data, delta_f_over_f, \
    filter_low_pass
from config import Config

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'select_good_sweeps.py'

Select good recordings (sweeps) that should be used for further analysis.
For these selected recordings convert the raw F values to delta F over F.

Result: 
- Delta F over F traces: /data/df_f_data.csv
- Labels: /data/df_f_data_labels.csv'
...

Requirements:
- The raw data: /data/raw_data.csv
- Metadata: /meta_data/meta_data.csv'
- Selection: /PrDA_good_sweeps.csv'

Nils Brehm  -  2025
"""


def test_plot(raw_f, df_f):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(raw_f, 'k')
    ax[1].plot(df_f, 'k')
    ax[0].plot([500, 500], [np.min(raw_f), np.max(raw_f)], 'r--')
    ax[1].plot([500, 500], [0, np.max(df_f)], 'r--')
    plt.show()


def select_sweeps(ca_file, sweeps_file):
    ca_data = load_raw_ca_data_only(ca_file)
    ca_labels = load_ca_data_headers_only(ca_file)
    sweeps = pd.read_csv(sweeps_file)
    sweep_names = sweeps['sweep_name']

    # Select "good" sweeps (info about good sweeps is found in "PrDA_good_sweeps.csv"
    idx = ca_labels.iloc[1, :].isin(sweep_names)
    selected_data = ca_data.loc[:, idx].dropna().reset_index(drop=True)

    # Get the selected labels
    selected_labels = ca_labels.loc[:, idx].dropna().reset_index(drop=True)
    return selected_data, selected_labels


def main():
    base_dir = Config.BASE_DIR
    print(f'\n ==== BASE DIR set to: {base_dir} ==== \n')

    ca_file = f'{base_dir}/data/ca_data.csv'
    meta_data_file = f'{base_dir}/meta_data/meta_data.csv'
    sweeps_file = f'{base_dir}/PrDA_good_sweeps.csv'  # only the column "sweep_name" is important!
    meta_data = load_ca_meta_data(meta_data_file)

    # ++++ DATA SELECTION ++++
    # Select "good" sweeps (info about good sweeps is found in "PrDA_good_sweeps.csv"
    selected_data, selected_labels = select_sweeps(ca_file, sweeps_file)

    # ++++ DELTA F OVER F ++++
    # Get Sampling Rate
    sampling_rate = meta_data['sampling_rate'][meta_data['name'] == 'sw_10'].item()

    # Compute dF/F using a sliding window (4 min)
    df_f, fbs = delta_f_over_f(selected_data, fr=sampling_rate, fbs_per=10, window=120)

    # Low and High Pass Filter before computing dF/F
    fil_lp = filter_low_pass(df_f, cutoff=0.5, fs=sampling_rate, order=2)
    # fil = filter_high_pass(fil_lp, cutoff=0.01, fs=sampling_rate, order=2)

    df_f = pd.DataFrame(fil_lp, columns=selected_data.keys())

    # Store to HDD
    df_f.to_csv(f'{base_dir}/data/df_f_data.csv', index=False)
    selected_labels.to_csv(f'{base_dir}/data/df_f_data_labels.csv', index=False)

    print('Data selected and converted to df/f')


if __name__ == '__main__':
    main()
