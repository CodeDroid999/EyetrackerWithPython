"""DatasetPaths module."""
from __future__ import annotations

from pathlib import Path


class DatasetPaths:
    """Defines the paths of a dataset."""

    def __init__(
            self,
            *,
            root: str | Path = 'data',
            dataset: str | None = None,
            raw: str = 'raw',
            events: str = 'events',
            preprocessed: str = 'preprocessed',
            downloads: str = 'downloads',
    ):
        """Initialize a set of dataset paths."""
        self._root = Path(root)
        self._dataset = dataset
        self._raw = raw
        self._events = events
        self._preprocessed = preprocessed
        self._downloads = downloads

    def get_preprocessed_filepath(
            self,
            raw_filepath: Path,
            *,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Path:
        """Get preprocessed filepath in accordance to filepath of the raw file."""
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw)

        if preprocessed_dirname is None:
            preprocessed_rootpath = self.preprocessed
        else:
            preprocessed_rootpath = self.dataset / preprocessed_dirname

        preprocessed_file_dirpath = preprocessed_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        preprocessed_filename = raw_filepath.stem + '.' + extension

        return preprocessed_file_dirpath / preprocessed_filename

    def raw_to_event_filepath(
            self,
            raw_filepath: Path,
            *,
            events_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Path:
        """Get event filepath in accordance to filepath of the raw file."""
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw)

        if events_dirname is None:
            events_rootpath = self.events
        else:
            events_rootpath = self.dataset / events_dirname

        events_file_dirpath = events_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        events_filename = raw_filepath.stem + '.' + extension

        return events_file_dirpath / events_filename

    @property
    def root(self) -> Path:
        """The root path to your dataset."""

        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        dataset.paths.root  # doctest: +SKIP
        Path('/path/to/your/dataset')

        return self._root

    @property
    def dataset(self) -> Path:
        """The path to the dataset directory"""
        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        dataset.path  # doctest: +SKIP
        Path('/path/to/your/dataset')

        if self._dataset is None:
            return self._root
        return self._root / self._dataset

    def fill_name(self, name: str) -> None:
        """Fill dataset directory name with dataset name."""
        if self._dataset is None:
            self._dataset = name

    @property
    def events(self) -> Path:
        """The path to the directory of the event data."""
        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        dataset.paths.events  # doctest: +SKIP
        Path('/path/to/your/dataset/events')
        return self.dataset / self._events

    @property
    def preprocessed(self) -> Path:
        """The path to the directory of the preprocessed gaze data."""
        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        dataset.paths.preprocessed  # doctest: +SKIP
        Path('/path/to/your/dataset/preprocessed')

        return self.dataset / self._preprocessed

    @property
    def raw(self) -> Path:
        """The path to the directory of the raw data."""
        import pytracker as pm
        dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        dataset.paths.raw  # doctest: +SKIP
        Path('/path/to/your/dataset/raw')


return self.dataset / self._raw


@property
def downloads(self) -> Path:    
    """The path to the directory of the raw data."""
    import pytracker as pm
    dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
    dataset.paths.downloads# doctest: +SKIP
    Path('/path/to/your/dataset/downloads')
return self.dataset / self._downloads
