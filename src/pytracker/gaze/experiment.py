"""This module holds the Experiment class."""
from __future__ import annotations

import numpy as np

from pytracker.gaze import transforms
from pytracker.gaze.screen import Screen
from pytracker.utils import decorators


@decorators.auto_str
class Experiment:
    """
    Experiment class for holding experiment properties.
    """
    def __init__(
        self, screen_width_px: int, screen_height_px: int,
        screen_width_cm: float, screen_height_cm: float,
        distance_cm: float, origin: tuple[str, str], sampling_rate: float,
    ):
        """
        Initializes Experiment.
        """
        self.screen = Screen(
            width_px=screen_width_px,
            height_px=screen_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            distance_cm=distance_cm,
            origin=origin,
        )
        self.sampling_rate = sampling_rate

    def pos2vel(
        self,
        arr: list[float] | list[list[float]] | np.ndarray,
        method: str = 'smooth',
        **kwargs,
    ) -> np.ndarray:
        return transforms.pos2vel(
            arr=arr, sampling_rate=self.sampling_rate, method=method, **kwargs,
        )
