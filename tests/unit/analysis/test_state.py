"""Unit tests for state.py — in particular, the strict PluginState.__setattr__
contract (dynamic attribute assignment raises unless the name is a declared
dataclass field). This is the exact risk flagged during the biopro ->
karcytics_sdk migration: any field WesternBlotPanel/analyzers set on
AnalysisState must be declared, or it raises AttributeError at runtime.
"""

import pytest

from karcytics_plugins.western_blot.analysis.state import AnalysisState

pytestmark = pytest.mark.unit


class TestAnalysisState:
    def test_declared_fields_are_settable(self):
        state = AnalysisState()
        state.is_inverted = True
        state.rotation_angle = 12.5
        assert state.is_inverted is True

    def test_metadata_field_is_settable(self):
        # WesternBlotPanel.get_state() attaches wizard position here.
        state = AnalysisState()
        state.metadata = {"current_step": 2, "max_step": 3}
        assert state.metadata["current_step"] == 2

    def test_undeclared_attribute_raises(self):
        state = AnalysisState()
        with pytest.raises(AttributeError):
            state.some_made_up_field = 123

    def test_original_image_alias_round_trips_through_raw_image(self):
        state = AnalysisState()
        state.original_image = "sentinel"
        assert state.raw_image == "sentinel"
        assert state.original_image == "sentinel"

    def test_to_dict_serializes_declared_fields(self):
        state = AnalysisState(is_inverted=True, rotation_angle=5.0)
        data = state.to_dict()
        assert data["is_inverted"] is True
        assert data["rotation_angle"] == 5.0
