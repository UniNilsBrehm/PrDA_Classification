import numpy as np
import pandas as pd
from utils import load_hdf5_as_data_frame, calcium_impulse_response, create_regressors_from_binary
from IPython import embed


def create_vr_regressor_traces(vr_binaries, cif, delta=False, norm=False):
    sweeps = vr_binaries.keys().to_numpy()
    vr_reg_traces = pd.DataFrame()
    for sw in sweeps:
        vr_regs = create_regressors_from_binary(vr_binaries[sw], cif, delta=delta, norm=norm)
        vr_reg_traces[sw] = vr_regs
    return vr_reg_traces


def main():
    # Directories
    base_dir = 'D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging'
    vr_binaries_dir = f'{base_dir}/ventral_root/selected_sweeps_ventral_root_binaries.csv'
    ca_sampling_rate_file = f'{base_dir}/meta_data/sampling_rate.csv'
    ca_df_f_file = f'{base_dir}/data/df_f_data.csv'

    # Load Data
    ca_sampling_rate = pd.read_csv(ca_sampling_rate_file)['sampling_rate'].item()

    # Detected VR Events for selected data set
    vr_binaries = pd.read_csv(vr_binaries_dir)

    # Create Calcium Impulse Response Function
    cif_t, cif = calcium_impulse_response(tau_rise=1, tau_decay=3, amplitude=1.0, sampling_rate=ca_sampling_rate, norm=True)

    # Create Regressors
    vr_reg_traces = create_vr_regressor_traces(vr_binaries, cif, delta=False, norm=False)
    vr_reg_traces_delta = create_vr_regressor_traces(vr_binaries, cif, delta=True, norm=False)

    # Store to HDD
    vr_reg_traces.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_regressor_traces.csv', index=False)
    vr_reg_traces_delta.to_csv(f'{base_dir}/ventral_root/selected_sweeps_ventral_root_regressor_traces_delta.csv', index=False)

    print('==== STORED REGRESSOR TRACES TO HDD ====')


if __name__ == '__main__':
    main()
