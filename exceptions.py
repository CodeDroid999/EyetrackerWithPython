"""Exceptions module."""
from __future__ import annotations


class InvalidProperty(Exception):
    """Raised if requested property is invalid."""

    def __init__(self, property_name: str, valid_properties: list[str]):
        message = f"property '{property_name}' is invalid. Valid properties are: {valid_properties}"
        super().__init__(message)
