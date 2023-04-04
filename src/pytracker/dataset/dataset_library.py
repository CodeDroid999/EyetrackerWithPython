"""DatasetLibrary module."""
from __future__ import annotations

from typing import Type
from typing import TypeVar

from pytracker.dataset.dataset_definition import DatasetDefinition


class DatasetLibrary:
    """Provides access by name to :py:class:`~pytracker.DatasetDefinition`."""

    definitions: dict[str, type[DatasetDefinition]] = {}

    @classmethod
    def add(cls, definition: type[DatasetDefinition]) -> None:
        """Add :py:class:`~pytracker.DatasetDefinition` to library."""
        cls.definitions[definition.name] = definition

    @classmethod
    def get(cls, name: str) -> type[DatasetDefinition]:
                return cls.definitions[name]


DatsetDefinitionClass = TypeVar('DatsetDefinitionClass', bound=Type[DatasetDefinition])


def register_dataset(cls: DatsetDefinitionClass) -> DatsetDefinitionClass:
    """Register a public dataset definition."""
    DatasetLibrary.add(cls)
    return cls
