"""This module provides the base dataset class."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from pytracker.dataset import dataset_download
from pytracker.dataset import dataset_files
from pytracker.dataset.dataset_definition import DatasetDefinition
from pytracker.dataset.dataset_library import DatasetLibrary
from pytracker.dataset.dataset_paths import DatasetPaths
from pytracker.events.event_processing import EventGazeProcessor
from pytracker.events.events import EventDataFrame
from pytracker.events.events import EventDetectionCallable
from pytracker.gaze import GazeDataFrame


class Dataset:
    """Dataset base class."""

    def __init__(
            self,
            definition: str | DatasetDefinition | type[DatasetDefinition],
            path: str | Path | DatasetPaths,
    ):
        """Initialize the dataset object."""
        self.fileinfo: pl.DataFrame = pl.DataFrame()
        self.gaze: list[GazeDataFrame] = []
        self.events: list[EventDataFrame] = []

        if isinstance(definition, str):
            definition = DatasetLibrary.get(definition)()
        if isinstance(definition, type):
            definition = definition()
        self.definition = deepcopy(definition)

        if isinstance(path, (str, Path)):
            self.paths = DatasetPaths(root=path, dataset='.')
        else:
            self.paths = deepcopy(path)
        # Fill dataset directory name with dataset definition name if specified.
        self.paths.fill_name(self.definition.name)

    def load(
            self,
            events: bool = False,
            preprocessed: bool = False,
            subset: None | dict[str, float | int | str | list[float | int | str]] = None,
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Parse file information and load all gaze files."""
        self.scan()
        self.fileinfo = dataset_files.take_subset(fileinfo=self.fileinfo, subset=subset)

        self.load_gaze_files(
            preprocessed=preprocessed, preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )

        if events:
            self.load_event_files(
                events_dirname=events_dirname,
                extension=extension,
            )

        return self

    def scan(self) -> Dataset:
        """Infer information from filepaths and filenames."""
        self.fileinfo = dataset_files.scan_dataset(definition=self.definition, paths=self.paths)
        return self

    def load_gaze_files(
            self,
            preprocessed: bool = False,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Dataset:
        """Load all available gaze data files."""
        self._check_fileinfo()
        self.gaze = dataset_files.load_gaze_files(
            definition=self.definition,
            fileinfo=self.fileinfo,
            paths=self.paths,
            preprocessed=preprocessed,
            preprocessed_dirname=preprocessed_dirname,
            extension=extension,
        )
        return self

    def load_event_files(
        self,
        events_dirname: str | None = None,
        extension: str = 'feather',
    ) -> Dataset:
        """Load all available event files."""
        self._check_fileinfo()
        self.events = dataset_files.load_event_files(
            definition=self.definition,
            fileinfo=self.fileinfo,
            paths=self.paths,
            events_dirname=events_dirname,
            extension=extension,
        )
        return self

    def pix2deg(self, verbose: bool = True) -> Dataset:
        """Compute gaze positions in degrees of visual angle from pixel coordinates."""
        self._check_gaze_dataframe()

        disable_progressbar = not verbose
        for gaze_df in tqdm(self.gaze, disable=disable_progressbar):
            gaze_df.pix2deg()

        return self

    def pos2vel(self, method: str = 'smooth', verbose: bool = True, **kwargs) -> Dataset:
        """Compute gaze velocites in dva/s from dva coordinates."""
        self._check_gaze_dataframe()

        disable_progressbar = not verbose
        for gaze_df in tqdm(self.gaze, disable=disable_progressbar):
            gaze_df.pos2vel(method=method, **kwargs)

        return self

    def detect_events(
            self,
            method: EventDetectionCallable,
            eye: str | None = 'auto',
            clear: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> Dataset:
        """Detect events by applying a specific event detection method."""
        self._check_gaze_dataframe()
        # Automatically infer eye to use for event detection.
        if eye == 'auto':
            if 'x_right_pos' in self.gaze[0].columns:
                eye = 'right'
            elif 'x_left_pos' in self.gaze[0].columns:
                eye = 'left'
            elif 'x_pos' in self.gaze[0].columns:
                eye = None
            else:
                raise AttributeError(
                    'Either right or left eye columns must be present in gaze data frame.'
                    f' Available columns are: {self.gaze[0].columns}',
                )

        if eye is None:
            position_columns = ['x_pos', 'y_pos']
            velocity_columns = ['x_vel', 'y_vel']
        else:
            position_columns = [f'x_{eye}_pos', f'y_{eye}_pos']
            velocity_columns = [f'x_{eye}_vel', f'y_{eye}_vel']

        if not set(position_columns).issubset(set(self.gaze[0].columns)):
            raise AttributeError(
                f'{eye} eye specified but required columns are not available in gaze dataframe.'
                f' required columns: {position_columns}'
                f', available columns: {self.gaze[0].columns}',
            )

        disable_progressbar = not verbose

        event_dfs: list[EventDataFrame] = []

        for gaze_df, fileinfo_row in tqdm(
                zip(self.gaze, self.fileinfo.to_dicts()), disable=disable_progressbar,
        ):

            positions = gaze_df.frame.select(position_columns).to_numpy()
            velocities = gaze_df.frame.select(velocity_columns).to_numpy()
            timesteps = gaze_df.frame.select('time').to_numpy()

            event_df = method(
                positions=positions, velocities=velocities, timesteps=timesteps, **kwargs,
            )

            event_df.frame = dataset_files.add_fileinfo(
                definition=self.definition,
                df=event_df.frame,
                fileinfo=fileinfo_row,
            )
            event_dfs.append(event_df)

        if not self.events or clear:
            self.events = event_dfs
            return self

        for file_id, event_df in enumerate(event_dfs):
            self.events[file_id].frame = pl.concat(
                [self.events[file_id].frame, event_df.frame],
                how='diagonal',
            )
        return self

    def compute_event_properties(
            self,
            event_properties: str | list[str],
            verbose: bool = True,
    ) -> Dataset:
        """Calculate an event property for and add it as a column to the event dataframe."""
        processor = EventGazeProcessor(event_properties)

        identifier_columns = [column for column in self.fileinfo.columns if column != 'filepath']

        disable_progressbar = not verbose
        for events, gaze in tqdm(zip(self.events, self.gaze), disable=disable_progressbar):
            new_properties = processor.process(events, gaze, identifiers=identifier_columns)

            new_properties = new_properties.drop(identifier_columns)
            new_properties = new_properties.drop(['name', 'onset', 'offset'])

            events.add_event_properties(new_properties)

        return self

    def clear_events(self) -> Dataset:
        """Clear event DataFrame."""
        if len(self.events) == 0:
            return self

        for file_id, _ in enumerate(self.events):
            self.events[file_id] = EventDataFrame()

        return self

    def save(
            self,
            events_dirname: str | None = None,
            preprocessed_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze and event files."""
        self.save_events(events_dirname, verbose=verbose, extension=extension)
        self.save_preprocessed(preprocessed_dirname, verbose=verbose, extension=extension)
        return self

    def save_events(
            self,
            events_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save events to files."""
        dataset_files.save_events(
            events=self.events,
            fileinfo=self.fileinfo,
            paths=self.paths,
            events_dirname=events_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def save_preprocessed(
            self,
            preprocessed_dirname: str | None = None,
            verbose: int = 1,
            extension: str = 'feather',
    ) -> Dataset:
        """Save preprocessed gaze files."""
        dataset_files.save_preprocessed(
            gaze=self.gaze,
            fileinfo=self.fileinfo,
            paths=self.paths,
            preprocessed_dirname=preprocessed_dirname,
            verbose=verbose,
            extension=extension,
        )
        return self

    def download(self, *, extract: bool = True, remove_finished: bool = False) -> Dataset:
        """Download dataset resources."""
        dataset_download.download_dataset(
            definition=self.definition,
            paths=self.paths,
            extract=extract,
            remove_finished=remove_finished,
        )
        return self

    def extract(self, remove_finished: bool = False) -> Dataset:
        """Extract downloaded dataset archive files"""
        dataset_download.extract_dataset(
            definition=self.definition,
            paths=self.paths,
            remove_finished=remove_finished,
        )
        return self

    @property
    def path(self) -> Path:
        """The path to the dataset directory."""
        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        dataset.path# doctest: +SKIP
        Path('/path/to/your/dataset')
        return self.paths.dataset

    def _check_fileinfo(self) -> None:
        """Check if fileinfo attribute is set and there is at least one row present."""
        if self.fileinfo is None:
            raise AttributeError(
                'fileinfo was not loaded yet. please run load() or scan() beforehand',
            )
        if len(self.fileinfo) == 0:
            raise AttributeError('no files present in fileinfo attribute')

    def _check_gaze_dataframe(self) -> None:
        """Check if gaze attribute is set and there is at least one gaze dataframe available."""
        if self.gaze is None:
            raise AttributeError('gaze files were not loaded yet. please run load() beforehand')
        if len(self.gaze) == 0:
            raise AttributeError('no files present in gaze attribute')
