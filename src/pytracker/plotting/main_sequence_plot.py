"""This module holds the main sequence plot."""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from polars import ColumnNotFoundError

from pytracker.events import EventDataFrame


def main_sequence_plot(
        event_df: EventDataFrame,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
) -> None:
    event_col_name = 'name'
    saccades = event_df.frame.filter(pl.col(event_col_name) == 'saccade')

    if saccades.is_empty():
        raise ValueError(
            'There are no saccades in the event dataframe. '
            'Please make sure you ran a saccade detection algorithm. '
            f'The event name should be stored in a colum called "{event_col_name}".',
        )

    try:
        peak_velocities = saccades['peak_velocity'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade peak velocities which are needed to create '
            'the main sequence plot. ',
        ) from exc

    try:
        amplitudes = saccades['amplitude'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade amplitudes which are needed to create '
            'the main sequence plot. ',
        ) from exc

    fig = plt.figure(figsize=figsize)

    plt.scatter(amplitudes, peak_velocities, color='purple', alpha=0.5)

    plt.title(title)
    plt.xlabel('Amplitude [dva]')
    plt.ylabel('Peak Velocity [dva/s]')

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()

    plt.close(fig)
