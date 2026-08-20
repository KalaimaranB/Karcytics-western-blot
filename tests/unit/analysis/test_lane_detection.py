"""Unit tests for lane_detection.py — pure numpy/scipy/skimage, no Qt."""

import numpy as np
import pytest

from karcytics_plugins.western_blot.analysis.lane_detection import (
    LaneROI,
    compute_vertical_projection,
    create_equal_lanes,
    detect_lanes_projection,
)

pytestmark = pytest.mark.unit


class TestLaneROI:
    def test_center_x_and_dimensions(self):
        lane = LaneROI(index=0, x_start=10, x_end=50, y_start=0, y_end=100)
        assert lane.center_x == 30
        assert lane.width == 40
        assert lane.height == 100

    def test_to_dict_from_dict_round_trip(self):
        lane = LaneROI(index=2, x_start=10, x_end=50, y_start=5, y_end=105)
        restored = LaneROI.from_dict(lane.to_dict())
        assert restored == LaneROI(index=2, x_start=10, x_end=50, y_start=5, y_end=105)

    def test_extract_returns_correct_region(self):
        image = np.arange(100).reshape(10, 10).astype(np.float64)
        lane = LaneROI(index=0, x_start=2, x_end=5, y_start=1, y_end=4)
        region = lane.extract(image)
        assert region.shape == (3, 3)
        np.testing.assert_array_equal(region, image[1:4, 2:5])


class TestComputeVerticalProjection:
    def test_output_shape_matches_width(self):
        image = np.random.rand(50, 120)
        projection = compute_vertical_projection(image)
        assert projection.shape == (120,)

    def test_uniform_image_gives_constant_projection(self):
        image = np.full((50, 120), 0.5)
        projection = compute_vertical_projection(image)
        np.testing.assert_allclose(projection, 0.5)


class TestCreateEqualLanes:
    def test_creates_requested_number_of_lanes(self):
        lanes = create_equal_lanes((200, 400), num_lanes=4)
        assert len(lanes) == 4
        assert [lane.index for lane in lanes] == [0, 1, 2, 3]

    def test_lanes_span_full_height(self):
        lanes = create_equal_lanes((200, 400), num_lanes=3)
        for lane in lanes:
            assert lane.y_start == 0
            assert lane.y_end == 200

    def test_lanes_are_contiguous_and_ordered(self):
        lanes = create_equal_lanes((200, 400), num_lanes=5)
        for prev, curr in zip(lanes, lanes[1:], strict=False):
            assert prev.x_end == curr.x_start

    def test_rejects_zero_lanes(self):
        with pytest.raises(ValueError):
            create_equal_lanes((200, 400), num_lanes=0)


class TestDetectLanesProjection:
    def test_returns_lane_list_for_banded_image(self, synthetic_blot_image):
        lanes = detect_lanes_projection(synthetic_blot_image, num_lanes=4)
        assert isinstance(lanes, list)
        for lane in lanes:
            assert isinstance(lane, LaneROI)

    def test_blank_image_does_not_raise(self, blank_image):
        # No bands present — should degrade gracefully rather than crash.
        detect_lanes_projection(blank_image)
