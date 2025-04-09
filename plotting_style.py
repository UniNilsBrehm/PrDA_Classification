import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict


def create_color_map(mode='grey'):

    if mode == 'green':
        colors = [
            (0.0, 0.0, 0.0),  # black
            (0.0, 1.0, 0.0)  # bright green
        ]

    elif mode == 'red':
        colors = [
            (0.0, 0.0, 0.0),  # black
            (1.0, 0.0, 0.0)  # bright
        ]

    elif mode == 'magenta':
        colors = [
            (0.0, 0.0, 0.0),  # black
            (1.0, 0.0, 1.0)  # bright magenta
        ]

    elif mode == 'corr':
        # Define the colormap using a dictionary with normalized values

        colors = [
            (0.0, 0.0, 1.0),  # blue
            (0.0, 0.0, 0.0),  # black
            (1.0, 0.0, 0.0)  # red
        ]

    elif mode == 'grey':
        colors = [
            (0.0, 0.0, 0.0),  # black
            # (0.2, 0.2, 0.2),  # dark
            # (0.6, 0.6, 0.6),  # medium
            (1.0, 1.0, 1.0)  # bright
        ]

    else:
        colors = [
            (0.0, 0.0, 0.0),  # black
            # (0.2, 0.2, 0.2),  # dark
            # (0.6, 0.6, 0.6),  # medium
            (1.0, 1.0, 1.0)  # bright
        ]

    custom_cmap = LinearSegmentedColormap.from_list('custom_green', colors, N=512)

    # Create the custom colormap.
    return custom_cmap


class PlotStyle:
    def __init__(self):
        self.line_styles = OrderedDict(
            [('solid', (0, ())),
             ('loosely dotted', (0, (1, 10))),
             ('dotted', (0, (1, 5))),
             ('densely dotted', (0, (1, 1))),

             ('loosely dashed', (0, (5, 10))),
             ('dashed', (0, (5, 5))),
             ('densely dashed', (0, (5, 1))),

             ('loosely dashdotted', (0, (3, 10, 1, 10))),
             ('dashdotted', (0, (3, 5, 1, 5))),
             ('densely dashdotted', (0, (3, 1, 1, 1))),

             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

        self.signal_color = 'black'

        # Line Styles
        self.lsStimulus = dict(color='tab:blue', lw=1.0, linestyle='-')
        self.lsCaTrace = dict(color='black', lw=1.0, linestyle='-')
        self.lsVrTrace = dict(color='gray', lw=1.0, linestyle='-')
        self.lsVrReg = dict(color='tab:red', lw=1.0, linestyle='-')
        self.lsVrReg_spontaneous = dict(color='tab:orange', lw=1.0, linestyle='-')
        self.lsStimulusReg = dict(color='tab:blue', lw=1.5, linestyle='-')
        self.lsStimulusReg_highlight = dict(color='tab:blue', lw=1.5, linestyle=self.line_styles['densely dotted'])
        self.lsStimulusReg_fading = dict(color='tab:blue', lw=1.0, linestyle='-', alpha=0.2)

        # Face Styles
        self.fsStimulusReg = dict(edgecolor='tab:blue', facecolor='tab:blue', alpha=1)
        self.fsVrReg = dict(edgecolor='tab:red', facecolor='tab:red', alpha=1)

        # Scatter Styles
        self.scStimulusOnset = dict(marker="v", markersize=5, markerfacecolor="tab:red", markeredgecolor="tab:red", linestyle="None")

        # Text Styles
        self.txtTimePoints = dict(color='white', fontsize=8)
        self.txtN = dict(color='black', fontsize=8)
        self.txtLabel = dict(color='black', fontsize=10)
        self.txtCaps = dict(color='black', fontsize=16)

        # Color maps and color bars
        self.cmap_default = 'afmhot'
        self.cmap_stimulus = create_color_map(mode='magenta')

        # Separation Lines
        self.sep_line = dict(color='r', linestyle='-',  lw=1.0)

        # Sub Fig Cap Text Size
        self.sub_fig_cap_text_size = 12
        self.sub_fig_text_color = 'black'
        self.sub_fig_cap_upper_case = True
        self.subfig_caps_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

        # MATPLOTLIB global settings:
        # Font:
        # matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['font.sans-serif'] = 'Arial'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.size'] = 8

        # Ticks:
        matplotlib.rcParams['xtick.major.pad'] = '2'
        matplotlib.rcParams['ytick.major.pad'] = '2'
        matplotlib.rcParams['ytick.major.size'] = 4
        matplotlib.rcParams['xtick.major.size'] = 4

        # Title Size:
        matplotlib.rcParams['axes.titlesize'] = 8

        # Axes Label Size:
        matplotlib.rcParams['axes.labelsize'] = 8

        # Axes Line Width:
        matplotlib.rcParams['axes.linewidth'] = 1

        # Tick Label Size:
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8

        # Line Width:
        matplotlib.rcParams['lines.linewidth'] = 1
        matplotlib.rcParams['lines.color'] = 'k'

        # Marker Size:
        matplotlib.rcParams['lines.markersize'] = 2

        # Error Bars:
        matplotlib.rcParams['errorbar.capsize'] = 0

        # Legend Font Size:
        matplotlib.rcParams['legend.fontsize'] = 8

        # Set pcolor shading
        matplotlib.rcParams['pcolor.shading'] = 'auto'
