""" This module holds path specific funtions."""
from __future__ import annotations

import re
from pathlib import Path


def get_filepaths(
        path: str | Path,
        extension: str | list[str] | None = None,
        regex: re.Pattern | None = None,
) -> list[Path]:
    """
    Get filepaths from rootpath depending on extension or regular expression.
    Passing extension and regex is mutually exclusive."""
    if extension is not None and regex is not None:
        raise ValueError('extension and regex are mutually exclusive')

    if extension is not None and isinstance(extension, str):
        extension = [extension]

    path = Path(path)
    if not path.is_dir():
        return []

    filepaths = []
    for childpath in path.iterdir():
        if childpath.is_dir():
            filepaths.extend(get_filepaths(path=childpath, extension=extension, regex=regex))
        else:
            # if extension specified and not matching, continue to next
            if extension and childpath.suffix not in extension:
                continue
            # if regex specified and not matching, continue to next
            if regex and not regex.match(childpath.name):
                continue
            filepaths.append(childpath)
    return filepaths


def match_filepaths(
        path: str | Path,
        regex: re.Pattern,
        relative: bool = True,
        relative_anchor: Path | None = None,
) -> list[dict[str, str]]:
    """Traverse path and match regular expression. """
    path = Path(path)

    if not path.exists():
        raise ValueError(f'path does not exist (path = {path})')

    if not path.is_dir():
        raise ValueError(f'path must point to a directory (path = {path})')

    if relative and relative_anchor is None:
        relative_anchor = path

    match_dicts: list[dict[str, str]] = []
    for childpath in path.iterdir():
        if childpath.is_dir():
            recursive_results = match_filepaths(
                path=childpath, regex=regex,
                relative=relative, relative_anchor=relative_anchor,
            )
            match_dicts.extend(recursive_results)
        else:
            match = regex.match(childpath.name)
            if match is not None:
                match_dict = match.groupdict()

                filepath = childpath
                if relative:
                    assert relative_anchor is not None
                    filepath = filepath.relative_to(relative_anchor)

                match_dict['filepath'] = str(filepath)
                match_dicts.append(match_dict)
    return match_dicts
