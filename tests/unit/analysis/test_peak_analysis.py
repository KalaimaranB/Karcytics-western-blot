"""Unit tests for peak_analysis.py — pure numpy/scipy, no Qt."""

import numpy as np
import pytest

from karcytics_plugins.western_blot.analysis.peak_analysis import (
    DetectedBand,
    detect_peaks,
    linear_baseline,
    rolling_ball_baseline,
)

pytestmark = pytest.mark.unit


def _single_peak_profile(length: int = 200, peak_pos: int = 100, height: float = 1.0) -> np.ndarray:
    profile = np.zeros(length, dtype=np.float64)
    width = 10
    x = np.arange(length)
    profile += height * np.exp(-((x - peak_pos) ** 2) / (2 * width**2))
    return profile


class TestDetectedBand:
    def test_constructs_with_required_fields(self):
        band = DetectedBand(lane_index=0, band_index=0, position=50, peak_height=0.5)
        assert band.lane_index == 0
        assert band.raw_height == 0.0  # default
        assert band.selected is True  # default, per docstring


class TestRollingBallBaseline:
    def test_output_same_length_as_input(self):
        profile = _single_peak_profile()
        baseline = rolling_ball_baseline(profile, radius=20)
        assert baseline.shape == profile.shape

    def test_floor_mode_baseline_is_at_or_below_profile(self):
        profile = _single_peak_profile()
        baseline = rolling_ball_baseline(profile, radius=20, mode="floor")
        # Floor mode should never estimate a baseline above the raw signal.
        assert np.all(baseline <= profile + 1e-9)

    def test_flat_profile_gives_flat_baseline(self):
        profile = np.full(100, 0.3)
        baseline = rolling_ball_baseline(profile, radius=10)
        np.testing.assert_allclose(baseline, 0.3)


class TestLinearBaseline:
    def test_no_peaks_returns_zeros(self):
        profile = _single_peak_profile()
        baseline = linear_baseline(profile, peak_indices=np.array([], dtype=np.intp))
        np.testing.assert_array_equal(baseline, np.zeros_like(profile))

    def test_output_same_length_as_input(self):
        profile = _single_peak_profile()
        baseline = linear_baseline(profile, peak_indices=np.array([100], dtype=np.intp))
        assert baseline.shape == profile.shape


class TestDetectPeaks:
    def test_finds_single_synthetic_peak(self):
        profile = _single_peak_profile(peak_pos=100, height=1.0)
        peaks, _properties = detect_peaks(profile, min_peak_height=0.1, min_peak_distance=5)
        assert len(peaks) >= 1
        assert any(abs(p - 100) <= 2 for p in peaks)

    def test_flat_profile_finds_no_peaks(self):
        profile = np.zeros(100)
        peaks, _properties = detect_peaks(profile, min_peak_height=0.1, min_peak_distance=5)
        assert len(peaks) == 0
