"""This module provides an interface to the GazeBase dataset."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pytracker.dataset.dataset_definition import DatasetDefinition
from pytracker.dataset.dataset_library import register_dataset
from pytracker.gaze.experiment import Experiment


@dataclass
@register_dataset
class GazeBase(DatasetDefinition):
    """GazeBase dataset :cite:p:`GazeBase`."""
    import pytracker as pm
    dataset = pm.Dataset("GazeBase", path='data/GazeBase')
    dataset.download()
    dataset.load()
    pylint: disable=similarities
    name: str = 'GazeBase'

    mirrors: tuple[str] = (
        'https://figshare.com/raw_data/files/',
    )

    resources: tuple[dict[str, str]] = (
        {
            'resource': '27039812',
            'filename': 'raw_data.zip',
            'md5': 'cb7eb895fb48f8661decf038ab998c9a',
        },
    )

    experiment: Experiment = Experiment(
        screen_width_px=1680,
        screen_height_px=1050,
        screen_width_cm=47.4,
        screen_height_cm=29.7,
        distance_cm=55,
        origin='lower left',
        sampling_rate=1000,
    )

    filename_regex: str = (
        r'S_(?P<round_id>\d)(?P<subject_id>\d+)'
        r'_S(?P<session_id>\d+)'
        r'_(?P<task_name>.+).csv'
    )

    filename_regex_dtypes: dict[str, type] = field(
        default_factory=lambda: {
            'round_id': int,
            'subject_id': int,
            'session_id': int,
        },
    )

    column_map: dict[str, str] = field(
        default_factory=lambda: {
            'n': 'time',
            'x': 'x_left_pos',
            'y': 'y_left_pos',
            'val': 'validity',
            'xT': 'x_target_pos',
            'yT': 'y_target_pos',
        },
    )

    custom_read_kwargs: dict[str, Any] = field(default_factory=dict)
