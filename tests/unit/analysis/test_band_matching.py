"""Unit tests for band_matching.py — pure numpy/scipy, no Qt."""

import pytest

from karcytics_plugins.western_blot.analysis.band_matching import (
    assign_matched_bands,
    calculate_scientific_boundaries,
)

pytestmark = pytest.mark.unit


class TestAssignMatchedBands:
    def test_empty_input_returns_empty_mapping(self):
        assert assign_matched_bands(lane_to_band_positions={}, lane_to_shift={}) == {}

    def test_bands_within_tolerance_get_same_matched_id(self):
        mapping = assign_matched_bands(
            lane_to_band_positions={0: [(0, 100.0)], 1: [(0, 103.0)]},
            lane_to_shift={0: 0, 1: 0},
            tolerance_px=12.0,
        )
        assert mapping[(0, 0)][0] == mapping[(1, 0)][0]

    def test_bands_outside_tolerance_get_different_matched_ids(self):
        mapping = assign_matched_bands(
            lane_to_band_positions={0: [(0, 100.0)], 1: [(0, 300.0)]},
            lane_to_shift={0: 0, 1: 0},
            tolerance_px=12.0,
        )
        assert mapping[(0, 0)][0] != mapping[(1, 0)][0]

    def test_shift_is_applied_before_clustering(self):
        # Lane 1's band is at raw position 150, but a -50px shift brings it to
        # 100 — within tolerance of lane 0's band at 100.
        mapping = assign_matched_bands(
            lane_to_band_positions={0: [(0, 100.0)], 1: [(0, 150.0)]},
            lane_to_shift={0: 0, 1: -50},
            tolerance_px=12.0,
        )
        assert mapping[(0, 0)][0] == mapping[(1, 0)][0]


class TestCalculateScientificBoundaries:
    def test_identifies_missing_lanes(self):
        matched = assign_matched_bands(
            lane_to_band_positions={0: [(0, 100.0)], 1: [(0, 101.0)]},
            lane_to_shift={0: 0, 1: 0},
        )
        result = calculate_scientific_boundaries(
            lane_to_band_data={0: [(0, 100.0, 10.0)], 1: [(0, 101.0, 12.0)]},
            matched_mapping=matched,
            total_lanes=[0, 1, 2],
        )
        (consensus,) = result.values()
        assert consensus["present_in_lanes"] == {0, 1}
        assert consensus["missing_in_lanes"] == {2}

    def test_consensus_position_is_median(self):
        matched = assign_matched_bands(
            lane_to_band_positions={0: [(0, 98.0)], 1: [(0, 100.0)], 2: [(0, 102.0)]},
            lane_to_shift={0: 0, 1: 0, 2: 0},
        )
        result = calculate_scientific_boundaries(
            lane_to_band_data={
                0: [(0, 98.0, 10.0)],
                1: [(0, 100.0, 10.0)],
                2: [(0, 102.0, 10.0)],
            },
            matched_mapping=matched,
            total_lanes=[0, 1, 2],
        )
        (consensus,) = result.values()
        assert consensus["position"] == pytest.approx(100.0)
