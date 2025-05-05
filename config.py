import os


class Config:
    # Base directory where all project files are located
    BASE_DIR = os.path.expanduser('D:/WorkingData/PrTecDA_Data/PrDA_somas_Ca_imaging')
    # BASE_DIR = os.path.expanduser('D:/WorkingData/Initial_Data_PrDA_somas')
    # Define specific directories
    stimulus_protocols_dir: str = os.path.join(BASE_DIR, 'protocols')

    # Define specific files
    # Ventral Root
    vr_events_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_events.h5')
    vr_spontaneous_events_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_spontaneous_events.h5')
    vr_aligned_recordings_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_aligned_vr_recordings.h5')
    vr_binaries_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_binaries.csv')
    vr_all_binaries_file: str = os.path.join(BASE_DIR, 'ventral_root', 'selected_sweeps_ventral_root_all_binaries.h5')

    # Ca Imaging
    ca_labels_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_labels.csv')
    ca_df_f_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data.csv')
    ca_sampling_rate_file: str = os.path.join(BASE_DIR, 'meta_data', 'sampling_rate.csv')
    ca_df_f_sp_motor_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_spontaneous_motor.csv')
    ca_df_f_no_motor_file: str = os.path.join(BASE_DIR, 'data', 'df_f_data_no_motor.csv')

    # Plotting Data
    ca_responses_spontaneous_motor: str = os.path.join(BASE_DIR, 'figures', 'data',  'ca_responses_spontaneous_motor.csv')
    ca_responses_spontaneous_motor_baseline_corrected: str = os.path.join(BASE_DIR, 'figures', 'data',  'ca_responses_spontaneous_motor_baseline_corrected.csv')

    # Stimulus
    stimulus_traces_file: str = os.path.join(BASE_DIR, 'stimulus_traces', 'stimulus_traces.csv')

    # Linear Scoring
    linear_scoring_file: str = os.path.join(BASE_DIR, 'data', 'linear_scoring_results.csv')
