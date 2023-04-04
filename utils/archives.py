"""
Utils module for extracting archives and decompressing files.
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

from pytracker.utils.paths import get_filepaths


def extract_archive(
        source_path: Path,
        destination_path: Path | None = None,
        recursive: bool = True,
        remove_finished: bool = False,
) -> Path:
    """Extract an archive."""
    archive_type, compression_type = _detect_file_type(source_path)

    if not archive_type:
        return _decompress(
            source_path=source_path,
            destination_path=destination_path,
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key, since this was already done in__detect_file_type().
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    if destination_path is None:
        destination_path = source_path.parent

    # Extract file and remove archive if desired.
    extractor(source_path, destination_path, compression_type)
    if remove_finished:
        source_path.unlink()

    if recursive:
        # Get filepaths of all archives in extracted directory.
        archive_extensions = [
            *_ARCHIVE_EXTRACTORS.keys(),
            *_ARCHIVE_TYPE_ALIASES.keys(),
            *_COMPRESSED_FILE_OPENERS.keys(),
        ]
        archive_filepaths = get_filepaths(path=destination_path, extension=archive_extensions)

        # Extract all found archives.
        for archive_filepath in archive_filepaths:
            extract_destination = archive_filepath.parent / archive_filepath.stem

            extract_archive(
                source_path=archive_filepath,
                destination_path=extract_destination,
                recursive=recursive,
                remove_finished=remove_finished,
            )

    return destination_path


def _extract_tar(
        source_path: Path,
        destination_path: Path,
        compression: str | None,
) -> None:
    """Extract a tar archive."""
    with tarfile.open(source_path, f'r:{compression[1:]}' if compression else 'r') as archive:
        archive.extractall(destination_path)


def _extract_zip(
        source_path: Path,
        destination_path: Path,
        compression: str | None,
) -> None:
    """Extract a zip archive."""
    compression_id = _ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
    with zipfile.ZipFile(source_path, 'r', compression=compression_id) as archive:
        archive.extractall(destination_path)


_ARCHIVE_EXTRACTORS: dict[str, Callable[[Path, Path, str | None], None]] = {
    '.tar': _extract_tar,
    '.zip': _extract_zip,
}

_ARCHIVE_TYPE_ALIASES: dict[str, tuple[str, str]] = {
    '.tbz': ('.tar', '.bz2'),
    '.tbz2': ('.tar', '.bz2'),
    '.tgz': ('.tar', '.gz'),
}

_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = {
    '.bz2': bz2.open,
    '.gz': gzip.open,
    '.xz': lzma.open,
}

_ZIP_COMPRESSION_MAP: dict[str, int] = {
    '.bz2': zipfile.ZIP_BZIP2,
    '.xz': zipfile.ZIP_LZMA,
}


def _detect_file_type(filepath: Path) -> tuple[str | None, str | None]:
    """Detect the archive type and/or compression of a file."""
    suffixes = filepath.suffixes

    if not suffixes:
        raise RuntimeError(
            f"File '{filepath}' has no suffixes that could be used to detect the archive type or"
            ' compression.',
        )

    # Get last suffix only.
    suffix = suffixes[-1]

    # Check if suffix is a known alias.
    if suffix in _ARCHIVE_TYPE_ALIASES:
        return _ARCHIVE_TYPE_ALIASES[suffix]

    # Check if suffix refers to an archive type.
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, None

    # Check if the suffix refers to a compression type.
    if suffix in _COMPRESSED_FILE_OPENERS:
        # Check if there are more than one suffix.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # Check if the second last suffix refers to an archive type.
            if suffix2 not in _ARCHIVE_EXTRACTORS:
                raise RuntimeError(
                    f"Unsupported archive type: '{suffix2}'.\n"
                    f"Supported suffixes are: '{sorted(set(_ARCHIVE_EXTRACTORS))}'.",
                )
            # We detected a compressed archive file (e.g. tar.gz).
            return suffix2, suffix

        # We detected a single compressed file not an archive.
        return None, suffix

    # Raise error as we didn't find a valid suffix.
    valid_suffixes = sorted(
        set(_ARCHIVE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS),
    )
    raise RuntimeError(
        f"Unsupported compression or archive type: '{suffix}'.\n"
        f"Supported suffixes are: '{valid_suffixes}'.",
    )


def _decompress(
        source_path: Path,
        destination_path: Path | None = None,
        remove_finished: bool = False,
) -> Path:
    r"""Decompress a file."""
    _, compression = _detect_file_type(source_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {source_path.suffix}.")

    if destination_path is None:
        destination_path = source_path.parent / source_path.stem

    # We don't need to check for a missing key, since this was already done in _detect_file_type().
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    # Decompress by reading from compressed file and writing to destination.
    with compressed_file_opener(source_path, 'rb') as rfh, open(destination_path, 'wb') as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        source_path.unlink()

    return destination_path
