"""
Heatmap module.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from pytracker.gaze import GazeDataFrame


def heatmap(
    gaze: GazeDataFrame,
    position_columns: tuple[str, str] = ('x_pix', 'y_pix'),
    gridsize=(10, 10),
    cmap: colors.Colormap | str = 'jet',
    interpolation: str = 'gaussian',
    origin: str = 'lower',
    figsize: tuple[float, float] = (15, 10),
    cbar_label: str | None = None,
    show_cbar: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
    savepath: str | None = None,
) -> plt.Figure:
    # Extract x and y positions from the gaze dataframe
    x = gaze.frame[position_columns[0]]
    y = gaze.frame[position_columns[1]]

    # Check if experiment properties are available
    if not gaze.experiment:
        raise ValueError(
            'Experiment property of GazeDataFrame is None. '
            'GazeDataFrame must be associated with an experiment.',
        )

    # Get experiment screen properties
    screen = gaze.experiment.screen

    # Use screen properties to define the grid or degrees of visual angle
    if 'pix' in position_columns[0] and 'pix' in position_columns[1]:
        xmin, xmax = 0, screen.width_px
        ymin, ymax = 0, screen.height_px
    else:
        xmin, xmax = int(screen.x_min_dva), int(screen.x_max_dva)
        ymin, ymax = int(screen.y_min_dva), int(screen.y_max_dva)

    # Define the grid and bin the gaze data
    x_bins = np.linspace(xmin, xmax, num=gridsize[0]).astype(int)
    y_bins = np.linspace(ymin, ymax, num=gridsize[1]).astype(int)

    # Bin the gaze data
    heatmap_value, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Transpose to match the orientation of the screen
    heatmap_value = heatmap_value.T

    # Convert heatmap values from sample count to seconds
    heatmap_value /= gaze.experiment.sampling_rate

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    heatmap_plot = ax.imshow(
        heatmap_value,
        cmap=cmap,
        origin=origin,
        interpolation=interpolation,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )

    # Set the plot title and axis labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add a color bar to the plot
    if show_cbar:
        cbar = fig.colorbar(heatmap_plot, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    # Show or save the plot
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()

    return fig
