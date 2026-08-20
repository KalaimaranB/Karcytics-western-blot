"""Lane alignment and cross-lane band matching.

The goal is to make "corresponding bands" comparable across lanes even when
peak counts differ due to noise or low contrast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class LaneAlignment:
    """Per-lane vertical shift alignment (pixels)."""

    lane_index: int
    shift_px: int
    score: float


def _prepare_profile_for_alignment(
    profile: NDArray[np.float64],
    baseline: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    x = profile.astype(np.float64, copy=False)
    if baseline is not None and len(baseline) == len(x):
        x = np.maximum(x - baseline, 0.0)
    # Smooth to emphasize band structure over pixel noise
    x = uniform_filter1d(x, size=9)
    # Normalize for correlation robustness
    x = x - float(np.median(x))
    denom = float(np.linalg.norm(x))
    if denom > 0:
        x = x / denom
    return x


def align_lanes_by_correlation(
    profiles: list[NDArray[np.float64]],
    baselines: Sequence[NDArray[np.float64] | None] | None = None,
    *,
    ref_lane: int = 0,
    max_shift_px: int = 40,
    lane_indices: list[int] | None = None,
) -> dict[int, LaneAlignment]:
    """Estimate per-lane vertical shifts via cross-correlation.

    Args:
        profiles: Lane profiles (top-to-bottom 1D arrays).
        baselines: Optional baseline arrays (same lengths as profiles).
        ref_lane: Lane index to align others to.
        max_shift_px: Search window (+/-) in pixels.
        lane_indices: If provided, restrict alignment to these lane indices.

    Returns:
        Mapping lane_index -> LaneAlignment (shift in pixels).
        shift_px is added to a band position to get aligned_position.
    """
    if not profiles:
        return {}

    if lane_indices is None:
        lane_indices = list(range(len(profiles)))

    if baselines is None or len(baselines) != len(profiles):
        baselines = [None] * len(profiles)

    ref_lane = int(ref_lane)
    ref = _prepare_profile_for_alignment(profiles[ref_lane], baselines[ref_lane])
    n = len(ref)

    out: dict[int, LaneAlignment] = {}
    for li in lane_indices:
        li = int(li)
        x = _prepare_profile_for_alignment(profiles[li], baselines[li])
        # Ensure same length for correlation
        m = min(len(x), n)
        if m <= 5:
            out[li] = LaneAlignment(lane_index=li, shift_px=0, score=0.0)
            continue
        xr = x[:m]
        rr = ref[:m]

        best_shift = 0
        best_score = -1e9
        for s in range(-int(max_shift_px), int(max_shift_px) + 1):
            if s < 0:
                a = xr[-s:]
                b = rr[: len(a)]
            elif s > 0:
                a = xr[: m - s]
                b = rr[s : s + len(a)]
            else:
                a = xr
                b = rr
            if len(a) < 5:
                continue
            score = float(np.dot(a, b))
            if score > best_score:
                best_score = score
                best_shift = s

        out[li] = LaneAlignment(lane_index=li, shift_px=int(best_shift), score=float(best_score))

    return out


def assign_matched_bands(
    *,
    lane_to_band_positions: dict[int, list[tuple[int, float]]],
    lane_to_shift: dict[int, int],
    tolerance_px: float = 12.0,
) -> dict[tuple[int, int], tuple[int, float]]:
    """Assign a consistent matched_band id across lanes.

    Args:
        lane_to_band_positions: Mapping lane -> list of (band_local_index, position_px).
        lane_to_shift: Mapping lane -> vertical shift (added to position).
        tolerance_px: Max distance to merge peaks into the same matched band.

    Returns:
        Mapping (lane_index, band_local_index) -> (matched_band_id, aligned_position).
    """
    all_items: list[tuple[int, int, float]] = []
    for lane, bands in lane_to_band_positions.items():
        shift = float(lane_to_shift.get(lane, 0))
        for band_local_idx, pos in bands:
            all_items.append((lane, band_local_idx, float(pos) + shift))

    if not all_items:
        return {}

    all_items.sort(key=lambda t: t[2])

    clusters: list[list[tuple[int, int, float]]] = []
    current: list[tuple[int, int, float]] = [all_items[0]]
    for item in all_items[1:]:
        if abs(item[2] - current[-1][2]) <= float(tolerance_px):
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    mapping: dict[tuple[int, int], tuple[int, float]] = {}
    for matched_id, cluster in enumerate(clusters):
        # Cluster center for scoring
        center = float(np.median([p for _l, _b, p in cluster]))
        # If multiple peaks from same lane fall into one cluster, keep closest to center.
        per_lane: dict[int, tuple[int, float, float]] = {}
        for lane, band_local_idx, aligned_pos in cluster:
            dist = abs(aligned_pos - center)
            if lane not in per_lane or dist < per_lane[lane][2]:
                per_lane[lane] = (band_local_idx, aligned_pos, dist)
        for lane, (band_local_idx, aligned_pos, _dist) in per_lane.items():
            mapping[(lane, band_local_idx)] = (matched_id, float(aligned_pos))

    return mapping


def calculate_scientific_boundaries(
    lane_to_band_data: dict[int, list[tuple[int, float, float]]],
    matched_mapping: dict[tuple[int, int], tuple[int, float]],
    total_lanes: list[int],
) -> dict[int, dict]:
    """Calculate consensus positions/widths for each matched band group and identify missing lanes.

    Args:
        lane_to_band_data: Mapping lane_idx -> list of (band_local_idx, position, width).
        matched_mapping: Output from assign_matched_bands.
        total_lanes: List of valid sample lane indices.

    Returns:
        Mapping matched_band_id -> dict with consensus data.
    """
    clusters: dict[int, list[tuple[float, float]]] = {}
    lanes_with_band: dict[int, set[int]] = {}

    for lane_idx, data_list in lane_to_band_data.items():
        for band_local_idx, pos, width in data_list:
            match_info = matched_mapping.get((lane_idx, band_local_idx))
            if not match_info:
                continue

            matched_id, _ = match_info

            if matched_id not in clusters:
                clusters[matched_id] = []
                lanes_with_band[matched_id] = set()

            clusters[matched_id].append((float(pos), float(width)))
            lanes_with_band[matched_id].add(lane_idx)

    consensus_data = {}
    all_lanes_set = set(total_lanes)

    for matched_id, data in clusters.items():
        positions = [d[0] for d in data]
        widths = [d[1] for d in data]

        present = lanes_with_band[matched_id]
        missing = all_lanes_set - present

        consensus_data[matched_id] = {
            "position": float(np.median(positions)),
            "width": float(np.median(widths)),
            "present_in_lanes": present,
            "missing_in_lanes": missing,
        }

    return consensus_data
