"""Lane detection algorithms for gel and blot images.

This module provides algorithms to identify individual lanes in gel
electrophoresis and western blot images. Lanes are the vertical
columns where samples are loaded and separated.

The primary approach uses **vertical intensity projection** — computing
the mean intensity of each column to produce a 1-D profile. Lanes
appear as regions of lower intensity (darker bands), separated by
gaps of higher intensity (brighter background).

Design Notes:
    - All algorithms accept and return NumPy arrays.
    - Lane boundaries are represented as a list of ``(x_start, x_end)``
      tuples, where x values are column indices.
    - The module is analysis-type agnostic — it works for western blots,
      SDS-PAGE gels, and any similar banded image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d
from skimage.filters import threshold_otsu


@dataclass
class LaneROI:
    """Region of interest representing a single lane.

    Attributes:
        index: Zero-based lane number (left-to-right).
        x_start: Left column boundary (inclusive).
        x_end: Right column boundary (exclusive).
        y_start: Top row boundary (inclusive).
        y_end: Bottom row boundary (exclusive).
        center_x: Horizontal center of the lane.
    """

    index: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    lane_type: str = "Sample"

    def to_dict(self) -> dict:
        """Serializes the lane data to a JSON-safe dictionary."""
        return {
            "index": self.index,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "y_start": self.y_start,
            "y_end": self.y_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LaneROI:
        """Rebuilds the LaneROI object from a dictionary."""
        return cls(
            index=data.get("index", 0),
            x_start=data.get("x_start", 0),
            x_end=data.get("x_end", 0),
            y_start=data.get("y_start", 0),
            y_end=data.get("y_end", 0),
        )

    @property
    def center_x(self) -> int:
        """Horizontal center column of the lane."""
        return (self.x_start + self.x_end) // 2

    @property
    def width(self) -> int:
        """Width of the lane in pixels."""
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        """Height of the lane in pixels."""
        return self.y_end - self.y_start

    def extract(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract the lane region from an image.

        Args:
            image: Source image array.

        Returns:
            Cropped image region for this lane.
        """
        return image[self.y_start : self.y_end, self.x_start : self.x_end]


def compute_vertical_projection(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the vertical intensity projection of an image.

    The vertical projection is the mean intensity of each column,
    producing a 1-D signal of length equal to the image width.

    For a dark-on-light image (standard western blot), lanes will
    appear as dips (lower values) in the projection.

    Args:
        image: Grayscale float64 image.

    Returns:
        1-D array of mean column intensities, same length as image width.
    """
    return np.mean(image, axis=0)


def detect_lanes_projection(
    image: NDArray[np.float64],
    num_lanes: int | None = None,
    smoothing_window: int = 15,  # noqa: ARG001 — kept for API compatibility
    min_lane_gap_fraction: float = 0.02,  # noqa: ARG001 — kept for API compatibility
) -> list[LaneROI]:
    """Detect lane boundaries by isolating structural band cores.

    Algorithm:
        1. Use Otsu thresholding to create a binary mask of ONLY the dark bands.
        2. Project vertically: lanes become dense blocks of 1s, gaps are 0s.
        3. Find the exact horizontal center of each lane block.
        4. Draw lane boundaries exactly halfway between the centers.
    """
    h, w = image.shape[:2]

    # 1. Ignore pure white rotation padding
    valid_mask = image < 0.99
    valid_pixels = image[valid_mask]

    if len(valid_pixels) < 100:
        if num_lanes is not None:
            return create_equal_lanes(image.shape, num_lanes, margin_fraction=0.0)
        return [LaneROI(index=0, x_start=0, x_end=w, y_start=0, y_end=h)]

    # 2. Strict threshold to isolate ONLY dark band cores
    # Multiplying by 0.95 ensures we don't pick up faint background noise.
    try:
        thresh = threshold_otsu(valid_pixels) * 0.95
    except Exception:
        thresh = float(np.percentile(valid_pixels, 20))

    band_mask = (image < thresh) & valid_mask

    # 3. Project vertically: lanes become mountains of 1s, gaps/shadows are 0s
    profile = np.sum(band_mask, axis=0)

    # Smooth slightly to bridge micro-gaps within a single lane's width
    smooth_size = max(3, int(w * 0.01))
    profile = uniform_filter1d(profile, size=smooth_size)

    max_mass = float(np.max(profile))
    if max_mass < 1:  # Failsafe if image is completely blank
        if num_lanes is not None:
            return create_equal_lanes(image.shape, num_lanes, margin_fraction=0.0)
        return [LaneROI(index=0, x_start=0, x_end=w, y_start=0, y_end=h)]

    # 4. Filter out artifacts: A column is part of a lane if it has at least 10%
    # of the max lane's mass. This specifically ignores the far-left shadow.
    lane_mask = profile > (max_mass * 0.10)

    # 5. Extract continuous blocks (the lanes)
    in_lane = False
    start = 0
    blocks = []
    for i, val in enumerate(lane_mask):
        if val and not in_lane:
            in_lane = True
            start = i
        elif not val and in_lane:
            in_lane = False
            blocks.append((start, i))
    if in_lane:
        blocks.append((start, w))

    # Filter out tiny dust blocks
    min_width = max(2, int(w * 0.015))
    valid_blocks = [b for b in blocks if (b[1] - b[0]) >= min_width]

    if not valid_blocks:
        if num_lanes is not None:
            return create_equal_lanes(image.shape, num_lanes, margin_fraction=0.0)
        return [LaneROI(index=0, x_start=0, x_end=w, y_start=0, y_end=h)]

    # 6. Find the center X-coordinate of each detected lane
    lane_centers = [int((b[0] + b[1]) / 2) for b in valid_blocks]

    # If the user requested a specific number (e.g. they know there's a faint lane),
    # use the tallest peaks to fulfill that count.
    if num_lanes is not None:
        if len(lane_centers) > num_lanes:
            # More blocks found than requested — keep the most prominent ones
            masses = [np.sum(profile[b[0] : b[1]]) for b in valid_blocks]
            top_indices = np.argsort(masses)[-num_lanes:]
            lane_centers = np.sort(np.array(lane_centers)[top_indices]).tolist()
        elif len(lane_centers) < num_lanes:
            # Fewer blocks found than requested — trust the user, use equal spacing
            return create_equal_lanes(image.shape, num_lanes, margin_fraction=0.0)

    # 7. Build boundaries halfway between the lane centers
    # This provides a natural, centered crop for each lane.
    boundaries = [0]
    for i in range(len(lane_centers) - 1):
        mid = int((lane_centers[i] + lane_centers[i + 1]) / 2)
        boundaries.append(mid)
    boundaries.append(w)

    # 8. Construct the LaneROIs
    lanes = []
    for i in range(len(boundaries) - 1):
        lanes.append(
            LaneROI(
                index=i,
                x_start=boundaries[i],
                x_end=boundaries[i + 1],
                y_start=0,
                y_end=h,
            )
        )
    return lanes


def create_equal_lanes(
    image_shape: tuple[int, ...],
    num_lanes: int,
    margin_fraction: float = 0.02,
) -> list[LaneROI]:
    """Create equally-spaced lane ROIs.

    Used as a fallback when auto-detection fails, or when the user
    specifies the lane count and wants uniform spacing.

    Args:
        image_shape: Shape of the image ``(height, width, ...)``.
        num_lanes: Number of lanes to create.
        margin_fraction: Fraction of width to leave as margin on each side.

    Returns:
        List of ``LaneROI`` objects with equal widths.

    Raises:
        ValueError: If num_lanes < 1.
    """
    if num_lanes < 1:
        raise ValueError(f"num_lanes must be >= 1, got {num_lanes}")

    h, w = image_shape[:2]
    margin = int(w * margin_fraction)
    usable_width = w - 2 * margin
    lane_width = usable_width // num_lanes

    lanes = []
    for i in range(num_lanes):
        x_start = margin + i * lane_width
        x_end = margin + (i + 1) * lane_width if i < num_lanes - 1 else w - margin
        lanes.append(
            LaneROI(
                index=i,
                x_start=x_start,
                x_end=x_end,
                y_start=0,
                y_end=h,
            )
        )

    return lanes


def split_lane_at(
    lanes: list[LaneROI],
    split_x: int,
    min_lane_width: int = 10,
) -> list[LaneROI]:
    """Split a lane by inserting a new boundary at ``split_x``.

    Finds the lane that contains ``split_x`` and splits it into two
    lanes at that column.  All lanes are re-indexed left-to-right.

    Args:
        lanes: Existing list of ``LaneROI`` objects, ordered left-to-right.
        split_x: Column position where the new boundary should be placed.
        min_lane_width: Minimum width (px) for each resulting lane.
            If the split would create a lane narrower than this, the
            operation is rejected and the original list is returned.

    Returns:
        New list of ``LaneROI`` objects with one additional lane, or
        the original list unchanged if the split is invalid.
    """
    if not lanes:
        return lanes

    # Find the lane that contains split_x
    target_lane: LaneROI | None = None
    for lane in lanes:
        if lane.x_start < split_x < lane.x_end:
            target_lane = lane
            break

    if target_lane is None:
        return lanes  # Click was outside or exactly on an edge

    # Check minimum width constraint
    left_width = split_x - target_lane.x_start
    right_width = target_lane.x_end - split_x
    if left_width < min_lane_width or right_width < min_lane_width:
        return lanes  # Would create a too-narrow lane

    # Build the new lane list
    new_lanes = []
    idx = 0
    for lane in lanes:
        if lane is target_lane:
            # Left half
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=lane.x_start,
                    x_end=split_x,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1
            # Right half
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=split_x,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1
        else:
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=lane.x_start,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1

    return new_lanes


def merge_lanes_at(
    lanes: list[LaneROI],
    boundary_index: int,
) -> list[LaneROI]:
    """Merge two adjacent lanes by removing the boundary between them.

    The boundary at ``boundary_index`` separates lane ``boundary_index - 1``
    and lane ``boundary_index``.  The merged lane inherits the lane type
    of the left lane.

    Args:
        lanes: Existing list of ``LaneROI`` objects, ordered left-to-right.
        boundary_index: The internal boundary to remove (1-based: boundary 1
            separates lane 0 and lane 1, etc.).

    Returns:
        New list of ``LaneROI`` objects with one fewer lane, or
        the original list unchanged if the index is invalid.
    """
    if not lanes or len(lanes) < 2:
        return lanes

    if boundary_index < 1 or boundary_index >= len(lanes):
        return lanes  # Invalid boundary index

    left_lane = lanes[boundary_index - 1]
    right_lane = lanes[boundary_index]

    merged = LaneROI(
        index=0,  # Will be re-indexed below
        x_start=left_lane.x_start,
        x_end=right_lane.x_end,
        y_start=left_lane.y_start,
        y_end=left_lane.y_end,
        lane_type=left_lane.lane_type,
    )

    new_lanes = []
    idx = 0
    for i, lane in enumerate(lanes):
        if i == boundary_index - 1:
            merged.index = idx
            new_lanes.append(merged)
            idx += 1
        elif i == boundary_index:
            continue  # Skip — absorbed into merged
        else:
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=lane.x_start,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1

    return new_lanes


def insert_gap_at(
    lanes: list[LaneROI],
    gap_center_x: int,
    gap_width: int = 20,
    min_lane_width: int = 10,
) -> list[LaneROI]:
    """Insert an excluded gap region at ``gap_center_x``.

    Splits the lane containing ``gap_center_x`` into **three** parts:
    a left sample lane, a center gap lane (``lane_type="Exclude"``),
    and a right sample lane.  The gap lanes boundaries are regular
    ``LaneROI`` objects, so they are draggable via ``LaneBorderItem``
    in the canvas's lane-edit mode.

    Args:
        lanes: Existing list of ``LaneROI`` objects, ordered left-to-right.
        gap_center_x: Column position at the center of the gap.
        gap_width: Total width of the gap in pixels (default 20).
        min_lane_width: Minimum width (px) for each resulting lane.
            If the split would create a lane narrower than this, the
            operation is rejected and the original list is returned.

    Returns:
        New list of ``LaneROI`` objects with two additional lanes
        (the original lane becomes left + gap + right), or the
        original list unchanged if the gap doesn't fit.
    """
    if not lanes:
        return lanes

    half = gap_width // 2
    gap_left = gap_center_x - half
    gap_right = gap_center_x + half

    # Find the lane that contains the gap center
    target_lane: LaneROI | None = None
    for lane in lanes:
        if lane.x_start < gap_center_x < lane.x_end:
            target_lane = lane
            break

    if target_lane is None:
        return lanes

    # Check we have enough room for left_lane + gap + right_lane
    left_width = gap_left - target_lane.x_start
    right_width = target_lane.x_end - gap_right
    if left_width < min_lane_width or right_width < min_lane_width or gap_width < 4:
        return lanes

    # Build the new lane list
    new_lanes = []
    idx = 0
    for lane in lanes:
        if lane is target_lane:
            # Left part — keeps original lane type
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=lane.x_start,
                    x_end=gap_left,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1
            # Gap — auto-marked Exclude
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=gap_left,
                    x_end=gap_right,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type="Exclude",
                )
            )
            idx += 1
            # Right part — keeps original lane type
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=gap_right,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1
        else:
            new_lanes.append(
                LaneROI(
                    index=idx,
                    x_start=lane.x_start,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    lane_type=lane.lane_type,
                )
            )
            idx += 1

    return new_lanes
