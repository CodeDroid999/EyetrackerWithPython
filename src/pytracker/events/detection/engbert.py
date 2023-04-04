"""
This module holds the implementation for the Engbert microsaccades algorithm.
"""
from __future__ import annotations

from collections.abc import Sized

import numpy as np

from pytracker.events.events import EventDataFrame
from pytracker.gaze.transforms import consecutive
from pytracker.utils import checks
from pytracker.utils.filters import filter_candidates_remove_nans


def microsaccades(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        minimum_duration: int = 6,
        threshold: np.ndarray | tuple[float] | str = 'engbert2015',
        threshold_factor: float = 6,
        minimum_threshold: float = 1e-10,
        include_nan: bool = False,
) -> EventDataFrame:
    """Detect micro-saccades from velocity gaze sequence."""
    positions = np.array(positions)
    velocities = np.array(velocities)

    checks.check_shapes_positions_velocities(positions=positions, velocities=velocities)

    if timesteps is None:
        timesteps = np.arange(len(velocities), dtype=np.int64)
    timesteps = np.array(timesteps)
    checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    if isinstance(threshold, str):
        threshold = compute_threshold(velocities, method=threshold)
    else:
        if isinstance(threshold, Sized) and len(threshold) != 2:
            raise ValueError('threshold must be either string or two-dimensional')
        threshold = np.array(threshold)

    if (threshold < minimum_threshold).any():
        raise ValueError(
            'threshold does not provide enough variance as required by min_threshold'
            f' ({threshold} < {minimum_threshold})',
        )

    # Radius of elliptic threshold.
    radius = threshold * threshold_factor

    # If value is greater than 1, point lies outside the ellipse.
    candidate_mask = np.greater(np.sum(np.power(velocities / radius, 2), axis=1), 1)

    # Add nans to candidates if desired.
    if include_nan:
        candidate_mask = np.logical_or(candidate_mask, np.isnan(velocities).any(axis=1))

    # Get indices of true values in candidate mask.
    candidate_indices = np.where(candidate_mask)[0]

    # Get all saccade candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    # Remove leading and trailing nan values from candidates.
    if include_nan:
        candidates = filter_candidates_remove_nans(candidates=candidates, values=velocities)

    # Filter all candidates by minimum duration.
    candidates = [
        candidate for candidate in candidates
        if len(candidate) > 0
        and timesteps[candidate[-1]] - timesteps[candidate[0]] >= minimum_duration
    ]

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate_indices[0] for candidate_indices in candidates]].flatten()
    # Offset of each event candidate is last event in candidate indices.
    offsets = timesteps[[candidate_indices[-1] for candidate_indices in candidates]].flatten()

    # Create event dataframe from onsets and offsets.
    event_df = EventDataFrame(name='saccade', onsets=onsets, offsets=offsets)
    return event_df


def compute_threshold(arr: np.ndarray, method: str = 'engbert2015') -> np.ndarray:
    """Determine threshold by computing variation."""
    if method == 'std':
        thx = np.nanstd(arr[:, 0])
        thy = np.nanstd(arr[:, 1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(arr[:, 0] - np.nanmedian(arr[:, 0])))
        thy = np.nanmedian(np.absolute(arr[:, 1] - np.nanmedian(arr[:, 1])))

    elif method == 'engbert2003':
        thx = np.sqrt(
            np.nanmedian(np.power(arr[:, 0], 2)) - np.power(np.nanmedian(arr[:, 0]), 2),
        )
        thy = np.sqrt(
            np.nanmedian(np.power(arr[:, 1], 2)) - np.power(np.nanmedian(arr[:, 1]), 2),
        )

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(arr[:, 0] - np.nanmedian(arr[:, 0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(arr[:, 1] - np.nanmedian(arr[:, 1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(f'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])
