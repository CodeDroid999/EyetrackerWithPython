"""
Transforms module.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from pytracker.utils import checks


def pix2deg(
        arr: float | list[float] | list[list[float]] | np.ndarray,
        screen_px: float | list[float] | tuple[float, float] | np.ndarray,
        screen_cm: float | list[float] | tuple[float, float] | np.ndarray,
        distance_cm: float,
        origin: str,
) -> np.ndarray:
    """Converts pixel screen coordinates to degrees of visual angle."""
    if arr is None:
        raise TypeError('arr must not be None')

    checks.check_no_zeros(screen_px, 'screen_px')
    checks.check_no_zeros(screen_cm, 'screen_px')
    checks.check_no_zeros(distance_cm, 'distance_cm')

    arr = np.array(arr)
    screen_px = np.array(screen_px)
    screen_cm = np.array(screen_cm)

    # Check basic arr dimensions.
    if arr.ndim not in [0, 1, 2]:
        raise ValueError(
            'Number of dimensions of arr must be either 0, 1 or 2'
            f' (arr.ndim: {arr.ndim})',
        )
    if arr.ndim == 2 and arr.shape[-1] not in [1, 2, 4]:
        raise ValueError(
            'Last coord dimension must have length 1, 2 or 4.'
            f' (arr.shape: {arr.shape})',
        )

    # check if arr dimensions match screen_px and screen_cm dimensions
    if arr.ndim in {0, 1}:
        if screen_px.ndim != 0 and screen_px.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_px is not')
        if screen_cm.ndim != 0 and screen_cm.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 2:
        if screen_px.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_px is not')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 4:
        if screen_px.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_px is not 2-dimensional')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_cm is not 2-dimensional')

        # We have binocular data. Double tile screen parameters.
        screen_px = np.tile(screen_px, 2)
        screen_cm = np.tile(screen_cm, 2)

    # Compute eye-to-screen-distance in pixels.
    distance_px = distance_cm * (screen_px / screen_cm)

    # If pixel coordinate system is not centered, shift pixel coordinate to the center.
    if origin == 'lower left':
        arr = arr - (screen_px - 1) / 2
    elif origin != 'center':
        raise ValueError(f'origin {origin} is not supported.')

    # 180 / pi transforms arc measure to degrees.
    return np.arctan2(arr, distance_px) * 180 / np.pi


def pos2vel(
        arr: list[float] | list[list[float]] | np.ndarray,
        sampling_rate: float = 1000,
        method: str = 'smooth',
        **kwargs,
) -> np.ndarray:
    if sampling_rate <= 0:
        raise ValueError('sampling_rate needs to be above zero')

    # make sure that we're operating on a numpy array
    arr = np.array(arr)

    if arr.ndim not in [1, 2]:
        raise ValueError(
            'arr needs to have 1 or 2 dimensions (are: {arr.ndim = })',
        )
    if method == 'smooth' and arr.shape[0] < 6:
        raise ValueError(
            'arr has to have at least 6 elements for method "smooth"',
        )
    if method == 'neighbors' and arr.shape[0] < 3:
        raise ValueError(
            'arr has to have at least 3 elements for method "neighbors"',
        )
    if method == 'preceding' and arr.shape[0] < 2:
        raise ValueError(
            'arr has to have at least 2 elements for method "preceding"',
        )
    if method != 'savitzky_golay' and kwargs:
        raise ValueError(
            'selected method doesn\'t support any additional kwargs',
        )

    N = arr.shape[0]
    v = np.zeros(arr.shape)

    valid_methods = ['smooth', 'neighbors', 'preceding', 'savitzky_golay']
    if method == 'smooth':
        # center is N - 2
        moving_avg = arr[4:N] + arr[3:N - 1] - arr[1:N - 3] - arr[0:N - 4]
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        v[2:N - 2] = moving_avg * sampling_rate / 6

        # for second and second last sample:
        # calculate vocity from preceding and subsequent sample
        v[1] = (arr[2] - arr[0]) * sampling_rate / 2
        v[N - 2] = (arr[N - 1] - arr[N - 3]) * sampling_rate / 2

        # for first and second sample:
        # calculate velocity from current and neighboring sample
        v[0] = (arr[1] - arr[0]) * sampling_rate / 2
        v[N - 1] = (arr[N - 1] - arr[N - 2]) * sampling_rate / 2

    elif method == 'neighbors':
        # window size is two, so we need to divide by two
        v[1:N - 1] = (arr[2:N] - arr[0:N - 2]) * sampling_rate / 2

    elif method == 'preceding':
        v[1:N] = (arr[1:N] - arr[0:N - 1]) * sampling_rate

    elif method == 'savitzky_golay':
        # transform to velocities
        if arr.ndim == 1:
            v = savgol_filter(x=arr, deriv=1, **kwargs)
        else:  # we already checked for error cases

            for channel_id in range(arr.shape[1]):
                v[:, channel_id] = savgol_filter(
                    x=arr[:, channel_id], deriv=1, **kwargs,
                )
        v = v * sampling_rate

    else:
        raise ValueError(
            f'Method needs to be in {valid_methods}'
            f' (is: {method})',
        )

    return v


def norm(arr: np.ndarray, axis: int | None = None) -> np.ndarray | Any:
    r"""Takes the norm of an array."""
    if axis is None:
        # for single vector and array of vectors the axis is 0
        # shape is assumed to be either (2, ) or (2, sequence_length)
        if arr.ndim in {1, 2}:
            axis = 0

        # for batched array of vectors, the
        # shape is assumed to be (n_batches, 2, sequence_length)
        elif arr.ndim == 3:
            axis = 1

        else:
            raise ValueError(
                f'Axis can not be inferred in case of more than 3 '
                f'input array dimensions (arr.shape={arr.shape}).'
                f' Either reduce the number of input array '
                f'dimensions or specify `axis` explicitly.',
            )

    return np.linalg.norm(arr, axis=axis)


def split(
        arr: np.ndarray,
        window_size: int,
        keep_padded: bool = True,
) -> np.ndarray:
    n, rest = np.divmod(arr.shape[1], window_size)

    if rest > 0 and keep_padded:
        n_rows = arr.shape[0] * (n + 1)
    else:
        n_rows = arr.shape[0] * n

    arr_split = np.nan * np.ones((n_rows, window_size, arr.shape[2]))

    idx = 0
    for arr_instance in arr:
        # Create an array of indicies where sequence will be split.
        split_indices = np.arange(start=window_size, stop=(n + 1) * window_size, step=window_size)

        # The length of the last element in list will be equal to rest.
        arr_split_list = np.split(arr_instance, split_indices)

        # Put the first n elements of split list into output array.
        arr_split[idx:idx + n] = arr_split_list[:-1]
        idx += n

        if rest > 0 and keep_padded:
            # Insert last split, remaining padded samples are already np.nan.
            arr_split[idx, :rest] = arr_split_list[-1]
            idx += 1

    return arr_split


def downsample(
        arr: np.ndarray,
        factor: int,
) -> np.ndarray:
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
