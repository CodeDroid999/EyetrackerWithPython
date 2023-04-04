"""DatasetDefinition module."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pytracker.gaze.experiment import Experiment


@dataclass
class DatasetDefinition:
    """
    Definition to initialize a :py:class:`~pytracker.Dataset`.
    """
    name: str = '.'

    mirrors: tuple[str, ...] = field(default_factory=tuple)

    resources: tuple[dict[str, str], ...] = field(default_factory=tuple)

    experiment: Experiment | None = None

    filename_regex: str = '.*'

    filename_regex_dtypes: dict[str, type] = field(default_factory=dict)

    custom_read_kwargs: dict[str, Any] = field(default_factory=dict)

    column_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.column_map) > 0:
            self.custom_read_kwargs['columns'] = list(self.column_map.keys())
            self.custom_read_kwargs['new_columns'] = list(self.column_map.values())
