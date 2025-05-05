import numpy as np
import pandas as pd
from utils import load_raw_ca_data_only, load_ca_meta_data, load_ca_data_headers_only, simple_interpolation
from config import Config
import matplotlib.pyplot as plt

"""
Dopaminergic Pretectum Somata Data 2024/2025  - Visual Stimulation Battery
==========================================================================
'match_sampling_rates.py'

Match all sampling rates to each other, so that in the end all recordings have the same  sampling rate.
Here this is hard coded. Following sweeps will be adjusted to the sampling rate of the other recordings:
['sw_01', 'sw_02', 'sw_03', 'sw_04']

Result: ca_data.csv (raw F values)

...

Requirements:
- The raw data: /data/raw_data.csv
- Metadata: /meta_data/meta_data.csv'

Nils Brehm  -  2025
"""


def adjust_frame_rate(data, old_fr, new_fr, test_plot=False):
    # Old Time Axis
    data_size = data.shape[0]
    max_t = data_size / old_fr

    old_time = np.linspace(0, max_t, data_size)

    # New Time Axis
    new_time = np.linspace(0, max_t, int(max_t * new_fr))  # Resampling over the same time duration

    # Interpolation to resample the first trace
    new_values = np.interp(new_time, old_time, data)

    # plot
    if test_plot:
        plt.plot(old_time, data, 'k.-')
        plt.plot(new_time, new_values, 'r.-')
        plt.show()

    return new_values


def match_sampling_rates(ca_data, ca_labels, sweeps, frame_rates):
    # Find columns with the desired sweep names in the second row (index=1)
    # Extract the data traces from row 5 onward
    selected_data = ca_data.loc[:, ca_labels.iloc[1, :].isin(sweeps)].iloc[5:].dropna().reset_index(drop=True)
    new_data = pd.DataFrame()
    i = 1
    for col in selected_data:
        print(f'*** {i} / {selected_data.shape[1]} ***')
        resampled_data = adjust_frame_rate(
            selected_data[col],
            old_fr=frame_rates['old'],
            new_fr=frame_rates['new'],
            test_plot=False
        )

        new_data[col] = resampled_data
        i += 1
    return new_data


def replace_data(ca_data, new_data, sweeps, ca_labels):
    # The recordings with the lower sampling rate a for some reason also ca. 4 s shorter...
    # So we need to pad these traces to match the sample count
    # Number of values to pad
    n = ca_data.shape[0] - new_data.shape[0]

    # Get the last value of each column
    last_values = new_data.iloc[-1]

    # Create a DataFrame with the same columns, filled with the last values
    padding = pd.DataFrame({col: [last_values[col]] * n for col in new_data.columns})

    # Concatenate the original DataFrame with the padding DataFrame
    new_data_padded = pd.concat([new_data, padding], ignore_index=True)

    # Update the original data frame (replace entries with the new padded ones)
    data = ca_data.copy()
    data.update(new_data_padded)

    return data


def align_sampling_rates(raw_data_file, meta_data_file):
    # Load the raw ca data ignoring the first 5 rows of the csv file containing metadata
    raw_ca_data = load_raw_ca_data_only(raw_data_file)

    # Load the metadata from the raw ca data file, but only the first 5 rows of the csv file containing metadata
    raw_ca_labels = load_ca_data_headers_only(raw_data_file)
    meta_data = load_ca_meta_data(meta_data_file)

    # For this Data Set, the sweeps 1 to 4 have a lower sampling rate than the rest of the sweeps
    selected_sweeps = ['sw_01', 'sw_02', 'sw_03', 'sw_04']
    old_fr = meta_data['sampling_rate'][meta_data['name'] == selected_sweeps[0]].item()
    new_fr = meta_data['sampling_rate'][meta_data['name'] == 'sw_10'].item()

    data_fr_adjusted = match_sampling_rates(raw_ca_data, raw_ca_labels, selected_sweeps, {'old': old_fr, 'new': new_fr})
    new_data_frame = replace_data(raw_ca_data, data_fr_adjusted, selected_sweeps, raw_ca_labels)

    # Add the header containing metadata back again
    new_data_set = pd.concat([raw_ca_labels, new_data_frame], ignore_index=True)

    return new_data_set


def main():
    base_dir = Config.BASE_DIR
    print(f'\n ==== BASE DIR set to: {base_dir} ==== \n')
    
    raw_data_file = f'{base_dir}/data/raw_data.csv'
    meta_data_file = f'{base_dir}/meta_data/meta_data.csv'
    save_dir = f'{base_dir}/data/ca_data.csv'

    result = align_sampling_rates(raw_data_file, meta_data_file)

    # Store it to HDD
    result.to_csv(save_dir, index=False, header=None)
    print('++++ Align Sampling Rates and stored adjusted data to HDD ++++')


if __name__ == '__main__':
    main()
