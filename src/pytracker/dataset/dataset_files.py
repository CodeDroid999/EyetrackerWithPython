"""Functionality to scan, load and save dataset files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from pytracker.dataset.dataset_definition import DatasetDefinition
from pytracker.dataset.dataset_paths import DatasetPaths
from pytracker.events.events import EventDataFrame
from pytracker.gaze.gaze_dataframe import GazeDataFrame
from pytracker.utils.paths import match_filepaths


def scan_dataset(definition: DatasetDefinition, paths: DatasetPaths) -> pl.DataFrame:
    """Infer information from filepaths and filenames.
    """
    # Get all filepaths that match regular expression.
    fileinfo_dicts = match_filepaths(
        path=paths.raw,
        regex=re.compile(definition.filename_regex),
        relative=True,
    )

    if len(fileinfo_dicts) == 0:
        raise RuntimeError(f'no matching files found in {paths.raw}')

    # Create dataframe from all fileinfo records.
    fileinfo_df = pl.from_dicts(data=fileinfo_dicts, infer_schema_length=1)
    fileinfo_df = fileinfo_df.sort(by='filepath')

    fileinfo_df = fileinfo_df.with_columns([
        pl.col(fileinfo_key).cast(fileinfo_dtype)
        for fileinfo_key, fileinfo_dtype in definition.filename_regex_dtypes.items()
    ])

    return fileinfo_df


def load_event_files(
        definition: DatasetDefinition,
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        events_dirname: str | None = None,
        extension: str = 'feather',
) -> list[EventDataFrame]:
    """Load all event files according to fileinfo dataframe.
    """
    event_dfs: list[EventDataFrame] = []

    # read and preprocess input files
    for fileinfo_row in tqdm(fileinfo.to_dicts()):
        filepath = Path(fileinfo_row['filepath'])
        filepath = paths.raw / filepath

        filepath = paths.raw_to_event_filepath(
            filepath, events_dirname=events_dirname,
            extension=extension,
        )

        if extension == 'feather':
            event_df = pl.read_ipc(filepath)
        elif extension == 'csv':
            event_df = pl.read_csv(filepath)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )

        # Add fileinfo columns to dataframe.
        event_df = add_fileinfo(
            definition=definition,
            df=event_df,
            fileinfo=fileinfo_row,
        )

        event_dfs.append(EventDataFrame(event_df))

    return event_dfs


def load_gaze_files(
        definition: DatasetDefinition,
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        preprocessed: bool = False,
        preprocessed_dirname: str | None = None,
        extension: str = 'feather',
) -> list[GazeDataFrame]:
    """Load all available gaze data files.
    """
    gaze_dfs: list[GazeDataFrame] = []

    # Read gaze files from fileinfo attribute.
    for fileinfo_row in tqdm(fileinfo.to_dicts()):
        filepath = Path(fileinfo_row['filepath'])
        filepath = paths.raw / filepath

        if preprocessed:
            filepath = paths.get_preprocessed_filepath(
                filepath, preprocessed_dirname=preprocessed_dirname,
                extension=extension,
            )

        if filepath.suffix == '.csv':
            if preprocessed:
                gaze_df = pl.read_csv(filepath)
            else:
                gaze_df = pl.read_csv(filepath, **definition.custom_read_kwargs)
        elif filepath.suffix == '.feather':
            gaze_df = pl.read_ipc(filepath)
        else:
            raise RuntimeError(f'data files of type {filepath.suffix} are not supported')

        # Add fileinfo columns to dataframe.
        gaze_df = add_fileinfo(
            definition=definition,
            df=gaze_df,
            fileinfo=fileinfo_row,
        )

        gaze_dfs.append(GazeDataFrame(gaze_df, experiment=definition.experiment))

    return gaze_dfs


def add_fileinfo(
        definition: DatasetDefinition,
        df: pl.DataFrame,
        fileinfo: dict[str, Any],
) -> pl.DataFrame:
    """Add columns from fileinfo to dataframe.
    """

    df = df.select(
        [
            pl.lit(value).alias(column)
            for column, value in fileinfo.items()
            if column != 'filepath'
        ] + [pl.all()],
    )

    # Cast columns from fileinfo according to specification.
    df = df.with_columns([
        pl.col(fileinfo_key).cast(fileinfo_dtype)
        for fileinfo_key, fileinfo_dtype in definition.filename_regex_dtypes.items()
    ])
    return df


def save_events(
        events: list[EventDataFrame],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        events_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save events to files.
    """
    disable_progressbar = not verbose

    for file_id, event_df in enumerate(tqdm(events, disable=disable_progressbar)):
        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        events_filepath = paths.raw_to_event_filepath(
            raw_filepath, events_dirname=events_dirname,
            extension=extension,
        )

        event_df_out = event_df.frame.clone()
        for column in event_df_out.columns:
            if column in fileinfo.columns:
                event_df_out = event_df_out.drop(column)

        if verbose >= 2:
            print('Save file to', events_filepath)

        events_filepath.parent.mkdir(parents=True, exist_ok=True)
        if extension == 'feather':
            event_df_out.write_ipc(events_filepath)
        elif extension == 'csv':
            event_df_out.write_csv(events_filepath)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def save_preprocessed(
        gaze: list[GazeDataFrame],
        fileinfo: pl.DataFrame,
        paths: DatasetPaths,
        preprocessed_dirname: str | None = None,
        verbose: int = 1,
        extension: str = 'feather',
) -> None:
    """Save preprocessed gaze files.
    """
    disable_progressbar = not verbose

    for file_id, gaze_df in enumerate(tqdm(gaze, disable=disable_progressbar)):
        raw_filepath = paths.raw / Path(fileinfo[file_id, 'filepath'])
        preprocessed_filepath = paths.get_preprocessed_filepath(
            raw_filepath, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        gaze_df_out = gaze_df.frame.clone()
        for column in gaze_df.columns:
            if column in fileinfo.columns:
                gaze_df_out = gaze_df_out.drop(column)

        if verbose >= 2:
            print('Save file to', preprocessed_filepath)

        preprocessed_filepath.parent.mkdir(parents=True, exist_ok=True)
        if extension == 'feather':
            gaze_df_out.write_ipc(preprocessed_filepath)
        elif extension == 'csv':
            gaze_df_out.write_csv(preprocessed_filepath)
        else:
            valid_extensions = ['csv', 'feather']
            raise ValueError(
                f'unsupported file format "{extension}".'
                f'Supported formats are: {valid_extensions}',
            )


def take_subset(
        fileinfo: pl.DataFrame,
        subset: None | dict[
            str, bool | float | int | str | list[bool | float | int | str],
        ] = None,
) -> pl.DataFrame:
    """Take a subset of the fileinfo dataframe"""
    if subset is None:
        return fileinfo

    if not isinstance(subset, dict):
        raise TypeError(f'subset must be of type dict but is of type {type(subset)}')

    for subset_key, subset_value in subset.items():
        if not isinstance(subset_key, str):
            raise TypeError(
                f'subset keys must be of type str but key {subset_key} is of type'
                f' {type(subset_key)}',
            )

        if subset_key not in fileinfo.columns:
            raise ValueError(
                f'subset key {subset_key} must be a column in the fileinfo attribute.'
                f' Available columns are: {fileinfo.columns}',
            )

        if isinstance(subset_value, (bool, float, int, str)):
            column_values = [subset_value]
        elif isinstance(subset_value, (list, tuple)):
            column_values = subset_value
        else:
            raise TypeError(
                f'subset value must be of type bool, float, int, str or a list of these but'
                f' key-value pair {subset_key}: {subset_value} is of type {type(subset_value)}',
            )

        fileinfo = fileinfo.filter(pl.col(subset_key).is_in(column_values))
    return fileinfo
