"""Module for event processing."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import polars as pl

from pytracker.events.event_properties import EVENT_PROPERTIES
from pytracker.events.events import EventDataFrame
from pytracker.exceptions import InvalidProperty
from pytracker.gaze.gaze_dataframe import GazeDataFrame


class EventProcessor:
    """Processes event and gaze dataframes."""

    def __init__(self, event_properties: str | list[str]):
        """Initialize processor with event property definitions."""
        if isinstance(event_properties, str):
            event_properties = [event_properties]

        for property_name in event_properties:
            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties = event_properties

    def process(self, events: EventDataFrame) -> pl.DataFrame:
        """Process event dataframe."""
        property_expressions: dict[str, Callable[[], pl.Expr]] = {
            property_name: EVENT_PROPERTIES[property_name]
            for property_name in self.event_properties
        }

        expression_list = [
            property_expression().alias(property_name)
            for property_name, property_expression in property_expressions.items()
        ]
        result = events.frame.select(expression_list)
        return result


class EventGazeProcessor:
    """Processes event and gaze dataframes. """

    def __init__(self, event_properties: str | list[str]):
        """Initialize processor with event property definitions."""
        if isinstance(event_properties, str):
            event_properties = [event_properties]

        for property_name in event_properties:
            if property_name not in EVENT_PROPERTIES:
                valid_properties = list(EVENT_PROPERTIES.keys())
                raise InvalidProperty(
                    property_name=property_name, valid_properties=valid_properties,
                )

        self.event_properties = event_properties

    def process(
            self,
            events: EventDataFrame,
            gaze: GazeDataFrame,
            identifiers: str | list[str],
    ) -> pl.DataFrame:
        """Process event and gaze dataframe."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        if len(identifiers) == 0:
            raise ValueError('list of identifiers must not be empty')

        property_expressions: dict[str, Callable[..., pl.Expr]] = {
            property_name: EVENT_PROPERTIES[property_name]
            for property_name in self.event_properties
        }

        property_kwargs: dict[str, dict[str, Any]] = {
            property_name: {} for property_name in property_expressions.keys()
        }
        for property_name, property_expression in property_expressions.items():
            property_args = inspect.getfullargspec(property_expression).args
            if 'velocity_columns' in property_args:
                velocity_columns = tuple(gaze.velocity_columns[:2])
                property_kwargs[property_name]['velocity_columns'] = velocity_columns

            if 'position_columns' in property_args:
                position_columns = tuple(gaze.position_columns[:2])
                property_kwargs[property_name]['position_columns'] = position_columns

        result = (
            gaze.frame.join(events.frame, on=identifiers)
            .filter(pl.col('time').is_between(pl.col('onset'), pl.col('offset')))
            .groupby([*identifiers, 'name', 'onset', 'offset'])
            .agg(
                [
                    property_expression(**property_kwargs[property_name])
                    .alias(property_name)
                    for property_name, property_expression in property_expressions.items()
                ],
            )
        )
        return result
