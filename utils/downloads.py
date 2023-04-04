"""
Utils module for downloading files.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from pytracker.utils.archives import extract_archive

USER_AGENT: str = '/pytracker'


def download_and_extract_archive(
        url: str,
        download_dirpath: Path,
        download_filename: str,
        extract_dirpath: Path | None = None,
        md5: str | None = None,
        recursive: bool = True,
        remove_finished: bool = False,
) -> None:
    """Download and extract archive file.
    """
    archive_path = download_file(
        url=url,
        dirpath=download_dirpath,
        filename=download_filename,
        md5=md5,
    )

    if extract_dirpath is None:
        extract_dirpath = download_dirpath

    print(f'Extracting {archive_path.name} to {extract_dirpath}')
    extract_archive(
        source_path=archive_path,
        destination_path=extract_dirpath,
        recursive=recursive,
        remove_finished=remove_finished,
    )


def download_file(
        url: str,
        dirpath: Path,
        filename: str,
        md5: str | None = None,
        max_redirect_hops: int = 3,
) -> Path:
    """Download a file from a URL and place it in root."""
    dirpath = dirpath.expanduser()
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / filename

    # check if file is already present locally
    if _check_integrity(filepath, md5):
        print('Using already downloaded and verified file:', filepath)
        return filepath
    print(f'Downloading {url} to {filepath}')

    # expand redirect chain if needed
    url = _get_redirected_url(url=url, max_hops=max_redirect_hops)

    # download the file
    try:
        _download_url(url=url, destination=filepath)

    except OSError as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Download failed. Trying https -> http instead.')
            print(f'Downloading {url} to {filepath}')
            _download_url(url=url, destination=filepath)
        else:
            raise e

    # check integrity of downloaded file
    if not _check_integrity(filepath=filepath, md5=md5):
        raise RuntimeError(f'File {filepath} not found or download corrupted.')

    return filepath


def _get_redirected_url(url: str, max_hops: int = 3) -> str:
    """Get redirected URL. """
    initial_url = url
    headers = {'Method': 'HEAD', 'User-Agent': USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    raise RuntimeError(
        f'Request to {initial_url} exceeded {max_hops} redirects.'
        f' The last redirect points to {url}.',
    )


class _DownloadProgressBar(tqdm):  # pylint: disable=inconsistent-mro
    """Progress bar for downloads
    """

    def __init__(self, **kwargs):
        super().__init__(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
      if tsize is not None:
            self.total = tsize
    return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def _download_url(url: str, destination: Path):
    """Download file from URL and save to destination."""
    with _DownloadProgressBar(desc=destination.name) as t:
        urllib.request.urlretrieve(url=url, filename=destination, reporthook=t.update_to)
        t.total = t.n


def _check_integrity(filepath: Path, md5: str | None = None) -> bool:
    """Check file integrity by MD5 checksum."""
    if not filepath.is_file():
        return False
    if md5 is None:
        return True

    # Calculate checksum and check for match.
    file_md5 = _calculate_md5(filepath)
    return file_md5 == md5


def _calculate_md5(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum."""
    file_md5 = hashlib.new('md5', usedforsecurity=False)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            file_md5.update(chunk)
    return file_md5.hexdigest()
