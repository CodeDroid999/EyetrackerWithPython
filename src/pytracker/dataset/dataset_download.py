"""This module provides the abstract base public dataset class."""
from __future__ import annotations

from urllib.error import URLError

from pytracker.dataset.dataset_definition import DatasetDefinition
from pytracker.dataset.dataset_paths import DatasetPaths
from pytracker.utils.archives import extract_archive
from pytracker.utils.downloads import download_file


def download_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        extract: bool = True,
        remove_finished: bool = False,
) -> None:
    """Download dataset resources.
    This downloads all resources of the dataset. Per default this also extracts all archives
    into :py:meth:`Dataset.paths.raw`,
        """
    if len(definition.mirrors) == 0:
        raise AttributeError('number of mirrors must not be zero to download dataset')

    if len(definition.resources) == 0:
        raise AttributeError('number of resources must not be zero to download dataset')

    paths.raw.mkdir(parents=True, exist_ok=True)

    for resource in definition.resources:
        success = False

        for mirror in definition.mirrors:

            url = f'{mirror}{resource["resource"]}'

            try:
                download_file(
                    url=url,
                    dirpath=paths.downloads,
                    filename=resource['filename'],
                    md5=resource['md5'],
                )
                success = True

            # pylint: disable=overlapping-except
            except (URLError, OSError, RuntimeError) as error:
                print(f'Failed to download (trying next):\n{error}')
                # downloading the resource, try next mirror
                continue

            # downloading the resource was successful, we don't need to try another mirror
            break

        if not success:
            raise RuntimeError(
                f"downloading resource {resource['resource']} failed for all mirrors.",
            )

    if extract:
        extract_dataset(
            definition=definition,
            paths=paths,
            remove_finished=remove_finished,
        )


def extract_dataset(
        definition: DatasetDefinition,
        paths: DatasetPaths,
        remove_finished: bool = False,
) -> None:
    """Extract downloaded dataset archive files.
       Remove archive files after extraction.
    """
    paths.raw.mkdir(parents=True, exist_ok=True)

    for resource in definition.resources:
        extract_archive(
            source_path=paths.downloads / resource['filename'],
            destination_path=paths.raw,
            recursive=True,
            remove_finished=remove_finished,
        )
