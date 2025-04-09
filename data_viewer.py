import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from utils import load_hdf5_as_dict, norm_min_max, calcium_impulse_response, create_regressors_from_binary
import pandas as pd
import seaborn as sns
from config import Config
from IPython import embed


class DataPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Scoring")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # window_geometry = "3200x800"
        window_geometry = f"{int(1920/1.2)}x{int(1080/2)}"
        self.root.geometry(window_geometry)
        self.base_dir = Config.BASE_DIR

        # Load data
        self.data = pd.read_csv(Config.ca_df_f_file)
        scores_file = pd.read_csv(Config.linear_scoring_file)
        # Remove multiple entries: Motor Spontaneous and Moving Target 01 and 02
        # There are mean values for these already present
        # idx = scores_file['reg'].isin(['motor_spontaneous', 'moving_target_01', 'moving_target_02', 'moving_target_01_motor', 'moving_target_02_motor'])
        idx = scores_file['reg'].isin(['motor_spontaneous'])
        self.scores = scores_file[~idx].reset_index(drop=True).pivot(index=['roi', 'sw'], columns='reg', values='score').reset_index()

        self.data_size = self.data.shape[0]
        self.data_labels = pd.read_csv(Config.ca_labels_file)
        self.ca_sampling_rate = pd.read_csv(Config.ca_sampling_rate_file)['sampling_rate'].item()
        self.data_time = np.linspace(0, self.data_size / self.ca_sampling_rate, self.data_size)
        self.stimulus = pd.read_csv(Config.stimulus_traces_file)
        self.motor_binaries = load_hdf5_as_dict(Config.vr_all_binaries_file)
        self.linear_scoring = pd.read_csv(Config.linear_scoring_file)

        # Generate Calcium Impulse Response Function
        _, self.cif = calcium_impulse_response(
            tau_rise=2, tau_decay=7, amplitude=1.0, sampling_rate=self.ca_sampling_rate, threshold=1e-5, norm=True
        )

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)  # Pack the toolbar above the canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Pack the canvas below the toolbar

        # start at random index
        # self.current_data_index = np.random.randint(0, self.data.shape[1])
        self.current_data_index = 0

        self.plot_data()

        self.root.bind("<Left>", lambda event: self.change_data(-1))
        self.root.bind("<Right>", lambda event: self.change_data(1))
        self.root.bind("q", lambda event: self.close())

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM)

        ttk.Button(self.button_frame, text="Previous", command=lambda: self.change_data(-1)).pack(side=tk.LEFT)
        ttk.Button(self.button_frame, text="Next", command=lambda: self.change_data(1)).pack(side=tk.LEFT)

    def close(self):
        self.root.quit()

    def plot_data(self):
        ca_trace = self.data.iloc[:, self.current_data_index]
        ca_trace_norm = norm_min_max(ca_trace)
        roi = ca_trace.name
        sw = self.data_labels[roi].iloc[1]
        ca_time = self.data_time
        stimulus_trace = self.stimulus[sw]
        scores = self.scores[self.scores['roi'] == int(roi)]
        motor_spontaneous_reg = create_regressors_from_binary(self.motor_binaries[sw]['spontaneous'], self.cif, delta=True, norm=True)
        motor_stimulus_events = self.motor_binaries[sw].drop(columns='spontaneous').sum(axis=1)
        motor_stimulus_reg = create_regressors_from_binary(motor_stimulus_events, self.cif, delta=True, norm=True)
        self.ax.clear()
        self.ax.set_title(f"ROI {roi} ({sw})", fontsize=16)

        self.ax.plot(ca_time, stimulus_trace, 'b', lw=1, label='Stimulus', alpha=1)
        self.ax.plot(ca_time, ca_trace_norm, 'k', lw=2, label='Ca Trace', alpha=1)
        self.ax.plot(ca_time, motor_spontaneous_reg, 'g', lw=1, label='Motor Spontaneous', alpha=1)
        self.ax.plot(ca_time, motor_stimulus_reg, 'r', lw=1, label='Motor Stimulus', alpha=1)

        # Show Scores
        s_types = [
            'moving_target_01', 'moving_target_02',
            'grating_appears', 'grating_0', 'grating_180', 'grating_disappears',
            'bright_loom', 'dark_loom', 'bright_flash', 'dark_flash'
        ]
        t_pos = [310, 375, 442, 505, 575, 640, 700, 760, 820, 885]
        self.ax.text(150, 1, f'spont. motor={scores["motor_spontaneous_MEAN"].item():.3f}', fontsize=10, color='tab:red')
        k = 0
        for s in s_types:
            self.ax.text(t_pos[k], 1, f'{scores[s].item():.3f}', fontsize=10, color='tab:blue')
            self.ax.text(t_pos[k], 0.95, f'{scores[f"{s}_motor"].item():.3f}', fontsize=10, color='tab:red')
            k += 1

        self.ax.legend(loc=2)

        sns.despine(fig=self.figure, left=True)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_xlabel('Time [s]')
        self.canvas.draw()

    def change_data(self, step):
        self.current_data_index = (self.current_data_index + step) % self.data.shape[1]
        self.plot_data()


def main():
    root = tk.Tk()
    app = DataPlotter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
