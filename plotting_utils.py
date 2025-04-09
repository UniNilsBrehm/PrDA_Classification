import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib import cm, colors, colorbar


def add_colorbar(ax, label, label_font_size, label_rotation, label_pad, cmap, v_min, v_max, ticks=None):
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    cb.ax.tick_params(labelsize=label_font_size)
    cb.set_label(label, rotation=label_rotation, labelpad=label_pad)
    if ticks is not None:
        cb.set_ticks(ticks)


def remove_y_axis(ax, ticks=True, label=True, complete=False):
    if complete:
        ax.yaxis.set_visible(False)  # Hide entire y-axis
    else:
        if ticks:
            ax.set_yticks([])  # Remove ticks
        if label:
            ax.set_ylabel('')  # Remove label


def remove_x_axis(ax, ticks=True, label=True, complete=False):
    if complete:
        ax.xaxis.set_visible(False)  # Hide entire y-axis
    else:
        if ticks:
            ax.set_xticks([])  # Remove ticks
        if label:
            ax.set_xlabel('')  # Remove label


def hide_axis_spines(ax, left=True, right=False, top=False, bottom=True):
    ax.spines['left'].set_visible(left)  # Hide left spine (y-axis line)
    ax.spines['right'].set_visible(right)  # Hide right spine
    ax.spines['top'].set_visible(top)  # Hide top spine (y-axis line)
    ax.spines['bottom'].set_visible(bottom)  # Hide bottom spine


def draw_sizebar(ax, size, label, loc, width, orientation='horizontal', font_size=10, color='black'):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.transforms import Bbox
    from matplotlib.font_manager import FontProperties

    if orientation == 'horizontal':
        h_size = size
        v_size = width
    elif orientation == 'vertical':
        h_size = width
        v_size = size
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

    asb = AnchoredSizeBar(ax.transData,
                          size=h_size,  # size of scale bar in x-axis units
                          label=label,
                          loc='center',  # the coordinate origin
                          pad=0.1,
                          borderpad=0.5,
                          sep=5,
                          color=color,
                          size_vertical=v_size,
                          frameon=False,
                          # bbox_to_anchor=Bbox.from_bounds(0, 0, 1, 1),
                          bbox_to_anchor=(loc[0], loc[1]),  # x, y
                          bbox_transform=ax.figure.transFigure,
                          fontproperties=FontProperties(size=font_size)
                          )
    ax.add_artist(asb)


def add_scale_bar(ax, size, label, location, orientation='horizontal', color='black', linewidth=2, fontsize=10,
                  padding=0.02):
    """
    Add a scale bar to a Matplotlib axis with support for relative coordinates.

    Parameters:
    - ax: Matplotlib axis where the scale bar will be added.
    - size: Length of the scale bar in data coordinates.
    - label: Label text for the scale bar.
    - location: Tuple (x, y) specifying the relative starting point of the scale bar (0 to 1 range).
    - orientation: 'horizontal' or 'vertical' for the scale bar's orientation.
    - color: Color of the scale bar.
    - linewidth: Thickness of the scale bar line.
    - fontsize: Font size of the label text.
    - padding: Padding (in relative coordinates) between the scale bar and the label.
    """
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert relative coordinates to data coordinates
    x_data = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_data = ylim[0] + location[1] * (ylim[1] - ylim[0])

    # Convert padding from relative to data units
    padding_x = padding * (xlim[1] - xlim[0])
    padding_y = padding * (ylim[1] - ylim[0])

    if orientation == 'horizontal':
        # Draw the horizontal scale bar
        ax.plot([x_data, x_data + size], [y_data, y_data], color=color, linewidth=linewidth)
        # Add the label centered below the bar
        ax.text(x_data + size / 2, y_data - padding_y, label, ha='center', va='top', fontsize=fontsize, color=color)
    elif orientation == 'vertical':
        # Draw the vertical scale bar
        ax.plot([x_data, x_data], [y_data, y_data + size], color=color, linewidth=linewidth)
        # Add the label to the right of the bar
        ax.text(x_data + padding_x, y_data + size / 2, label, ha='left', va='center', fontsize=fontsize, color=color)
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

