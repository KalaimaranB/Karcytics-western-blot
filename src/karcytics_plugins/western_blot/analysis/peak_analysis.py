"""Peak detection, baseline estimation, and densitometry calculations.

This module handles the signal-processing aspects of bio-image analysis:
extracting intensity profiles from lane regions, detecting peaks
(bands), estimating baselines, and computing integrated densities.

The algorithms are designed for real-world gel/blot images which have:
    - Uneven backgrounds and gradients.
    - Varying noise levels across the image.
    - Bands of widely different intensities and widths.
    - Large empty regions between bands.

Key improvements over naive peak detection:
    - **Baseline subtraction first**: Peaks are detected on the
      baseline-corrected signal, eliminating false positives from
      background gradients.
    - **Adaptive thresholding**: The noise floor is estimated from
      the corrected signal, and peaks must exceed a configurable
      multiple of the noise (signal-to-noise ratio).
    - **Width constraints**: Both minimum and maximum band widths
      are enforced, filtering out speckle noise and broad artifacts.
    - **Prominence filtering**: Peaks must stand out from their local
      surroundings, not just exceed an absolute threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks


@dataclass
class DetectedBand:
    """A single detected band within a lane.

    Attributes:
        lane_index: Zero-based index of the parent lane.
        band_index: Zero-based index of this band within the lane.
        position: Row position (pixel) of the band peak in the profile.
        peak_height: Intensity value at the peak maximum (above baseline).
        raw_height: Raw intensity value at the peak (before baseline subtraction).
        width: Estimated full-width-at-half-max of the band in pixels.
        integrated_intensity: Area under the peak above baseline.
        baseline_value: Estimated baseline intensity at this band position.
        snr: Signal-to-noise ratio of this band.
        selected: Whether this band is selected for analysis (user toggle).
    """

    lane_index: int
    band_index: int
    position: int
    peak_height: float
    raw_height: float = 0.0
    width: float = 0.0
    integrated_intensity: float = 0.0
    baseline_value: float = 0.0
    snr: float = 0.0
    selected: bool = True
    aligned_position: float | None = None
    matched_band: int | None = None
    match_score: float | None = None
    quality: float | None = None

    def to_dict(self) -> dict:
        """Serializes the band data to a JSON-safe dictionary."""
        return {
            "lane_index": self.lane_index,
            "band_index": self.band_index,
            "position": self.position,
            "peak_height": self.peak_height,
            "raw_height": self.raw_height,
            "width": self.width,
            "integrated_intensity": self.integrated_intensity,
            "baseline_value": self.baseline_value,
            "snr": self.snr,
            "selected": self.selected,
            "aligned_position": self.aligned_position,
            "matched_band": self.matched_band,
            "match_score": self.match_score,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedBand:
        """Rebuilds the DetectedBand object from a dictionary."""
        return cls(
            lane_index=data.get("lane_index", 0),
            band_index=data.get("band_index", 0),
            position=data.get("position", 0),
            peak_height=data.get("peak_height", 0.0),
            raw_height=data.get("raw_height", 0.0),
            width=data.get("width", 0.0),
            integrated_intensity=data.get("integrated_intensity", 0.0),
            baseline_value=data.get("baseline_value", 0.0),
            snr=data.get("snr", 0.0),
            selected=data.get("selected", True),
            aligned_position=data.get("aligned_position"),
            matched_band=data.get("matched_band"),
            match_score=data.get("match_score"),
            quality=data.get("quality"),
        )


def _band_likeness_score(
    lane_strip: NDArray[np.float64],
    peak_y: int,
    *,
    band_half_height: int = 1,
    bg_offset: int = 10,
    bg_half_height: int = 3,
) -> float:
    """Score how 'band-like' a peak is using local 2D structure.

    A real western band is horizontally coherent across a meaningful fraction
    of the lane width. Random whitespace wiggles tend to be weak and/or sparse.

    Returns:
        A non-negative score (higher = more band-like).
    """
    h, w = lane_strip.shape[:2]
    if w < 3 or h < 5:
        return 0.0

    y0 = int(np.clip(peak_y, 0, h - 1))

    # Band rows around the peak
    b_top = max(0, y0 - band_half_height)
    b_bot = min(h, y0 + band_half_height + 1)
    band_rows = lane_strip[b_top:b_bot, :]

    # Background rows a bit away from the peak (prefer below; fallback above)
    bg_center = y0 + bg_offset
    if bg_center + bg_half_height >= h:
        bg_center = y0 - bg_offset
    bg_top = max(0, bg_center - bg_half_height)
    bg_bot = min(h, bg_center + bg_half_height + 1)
    if bg_bot - bg_top < 1:
        return 0.0
    bg_rows = lane_strip[bg_top:bg_bot, :]

    band_line = np.median(band_rows, axis=0)
    bg_line = np.median(bg_rows, axis=0)

    # Contrast along x
    diff = band_line - bg_line
    # Require coherence: a decent fraction of x positions should be above a threshold
    robust_scale = float(np.median(np.abs(diff - np.median(diff)))) * 1.4826 + 1e-6
    thr = max(0.0, float(np.median(diff)) + 1.0 * robust_scale)
    frac = float(np.mean(diff > thr))

    # Score combines overall contrast and horizontal coverage
    contrast = float(np.percentile(diff, 75) - np.percentile(diff, 25))
    score = max(0.0, float(np.median(diff))) * (0.5 + frac) + contrast * frac
    return float(max(score, 0.0))


def extract_lane_profile(
    image: NDArray[np.float64],
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    sample_fraction: float = 0.5,
    statistic: str = "median",
) -> NDArray[np.float64]:
    """Extract a 1-D intensity profile along a lane.

    Takes the mean intensity across a centered strip within the lane
    (defaulting to the middle half), producing a profile from top to
    bottom that represents band densities.

    Note: The input image is expected to be already inverted by the
    preprocessing pipeline so that bands become peaks. This function
    simply extracts a robust 1-D summary across the lane width.

    Args:
        image: Grayscale float64 image in [0.0, 1.0].
        x_start: Left column of the lane (inclusive).
        x_end: Right column of the lane (exclusive).
        y_start: Top row of the lane (inclusive).
        y_end: Bottom row of the lane (exclusive).
        sample_fraction: Fraction of lane width to use for the profile.
            Centered in the lane. Default 0.5 (middle half) for better
            SNR than the ImageJ recommended 0.33.

    Returns:
        1-D intensity profile, length = (y_end - y_start).
        Higher values indicate darker bands.
    """
    lane_width = x_end - x_start
    center = (x_start + x_end) // 2
    sample_half_width = max(1, int(lane_width * sample_fraction / 2))

    strip_left = max(x_start, center - sample_half_width)
    strip_right = min(x_end, center + sample_half_width)

    lane_strip = image[y_start:y_end, strip_left:strip_right]

    # Robust aggregation across columns → 1-D profile along y-axis
    stat = statistic.lower().strip()
    if stat == "mean":
        profile = np.mean(lane_strip, axis=1)
    elif stat == "median":
        profile = np.median(lane_strip, axis=1)
    else:
        raise ValueError("statistic must be 'median' or 'mean'")

    return profile


def estimate_noise_level(
    profile: NDArray[np.float64],
    baseline: NDArray[np.float64],
) -> float:
    """Estimate the noise level in a baseline-corrected profile.

    Uses the median absolute deviation (MAD) of the corrected signal
    as a robust noise estimator. The MAD is less sensitive to outliers
    (i.e., actual peaks) than standard deviation.

    Args:
        profile: Raw intensity profile.
        baseline: Estimated baseline.

    Returns:
        Estimated noise standard deviation.
    """
    corrected = profile - baseline
    # MAD-based noise estimate (factor 1.4826 converts MAD to std)
    mad = float(np.median(np.abs(corrected - np.median(corrected))))
    noise = mad * 1.4826

    # Guard against zero noise (perfectly uniform image)
    return max(noise, 1e-6)


def orient_profile_for_bands(
    smoothed: NDArray[np.float64],
    baseline: NDArray[np.float64],
    force_valleys_as_bands: bool | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Orient a lane profile so that bands become positive peaks.

    Many real images may arrive in either polarity:
        - Bands as dark valleys below background (standard western blot:
          dark bands on white background).
        - Bands as bright peaks above background (fluorescent / inverted).

    This helper inspects the profile relative to its baseline and decides
    whether the dominant band-like structure is above or below the baseline.
    It then returns:
        - ``corrected``: Non-negative signal where bands are positive peaks.
        - ``display_profile``: baseline + corrected, suitable for plotting
          (bands always appear as upward peaks).
        - ``valleys_as_bands``: True if the original bands were valleys.
    """
    smoothed = np.asarray(smoothed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    if smoothed.shape != baseline.shape:
        raise ValueError("smoothed and baseline must have the same shape")

    delta = smoothed - baseline  # positive above baseline, negative at valleys

    if force_valleys_as_bands is not None:
        valleys_as_bands = bool(force_valleys_as_bands)
    else:
        max_pos = float(np.max(delta))
        max_neg = float(-np.min(delta))
        valleys_as_bands = max_neg >= max_pos * 0.8

    signal = -delta if valleys_as_bands else delta  # valleys become positive peaks

    corrected = np.maximum(signal, 0.0)

    # display_profile: baseline + corrected — bands always appear as peaks
    display_profile = baseline + corrected

    return corrected, display_profile, valleys_as_bands


def detect_peaks(
    profile: NDArray[np.float64],
    min_peak_height: float = 0.05,
    min_peak_distance: int = 10,
    min_prominence: float = 0.02,
    max_peak_width: float | None = None,
    min_peak_width: float = 2.0,
    plateau_size: tuple[int | None, int | None] | None = (1, None),
) -> tuple[NDArray[np.intp], dict]:
    """Detect peaks (bands) in a lane intensity profile.

    Uses ``scipy.signal.find_peaks`` with configurable parameters for
    height, distance, prominence, and width thresholds.

    Args:
        profile: 1-D intensity profile (should be baseline-corrected
            for best results).
        min_peak_height: Minimum absolute height for a peak.
        min_peak_distance: Minimum distance (pixels) between peaks.
        min_prominence: Minimum prominence for a peak.
        max_peak_width: Maximum allowed peak width in pixels. Peaks
            wider than this are rejected (likely artifacts). If None,
            no upper width limit is applied.
        min_peak_width: Minimum peak width in pixels. Peaks narrower
            than this are rejected (likely noise spikes).

    Returns:
        Tuple of (peak_indices, peak_properties).
    """
    width_range = (min_peak_width, max_peak_width) if max_peak_width else (min_peak_width, None)

    kwargs: dict[str, Any] = {
        "height": min_peak_height,
        "distance": min_peak_distance,
        "prominence": min_prominence,
        "width": width_range,
        "rel_height": 0.5,  # Measure width at half prominence
    }
    if plateau_size is not None:
        kwargs["plateau_size"] = plateau_size

    peaks, properties = find_peaks(profile, **kwargs)
    return peaks, properties


def rolling_ball_baseline(
    profile: NDArray[np.float64],
    radius: int = 50,
    mode: str = "floor",
) -> NDArray[np.float64]:
    """Estimate the baseline using the rolling ball algorithm.

    For images where bands are bright peaks (fluorescent), use mode='floor'
    (ball rolls under the curve, finding background minima).

    For images where bands are dark valleys (standard western blot), use
    mode='ceiling' (ball rolls over the curve, finding background maxima).

    Args:
        profile: 1-D intensity profile.
        radius: Radius of the rolling ball in pixels.
        mode: 'floor' (minimum filter) or 'ceiling' (maximum filter).

    Returns:
        Estimated baseline profile, same length as input.
    """
    from scipy.ndimage import maximum_filter1d

    size = 2 * radius + 1
    if mode == "ceiling":
        baseline = maximum_filter1d(profile, size=size)
    else:
        baseline = minimum_filter1d(profile, size=size)

    return uniform_filter1d(baseline, size=radius)


def linear_baseline(
    profile: NDArray[np.float64],
    peak_indices: NDArray[np.intp],
    window: int = 5,  # noqa: ARG001 — kept for API compatibility, see docstring
) -> NDArray[np.float64]:
    """Estimate baseline by linear interpolation between peak edges.

    For each peak, finds the local minima on either side and draws a
    straight line underneath.

    Args:
        profile: 1-D intensity profile.
        peak_indices: Indices of detected peaks.
        window: Search window size (unused, kept for API compatibility).

    Returns:
        Estimated baseline profile.
    """
    n = len(profile)

    if len(peak_indices) == 0:
        return np.zeros_like(profile)

    boundary_points = [0]
    for i in range(len(peak_indices)):
        peak = peak_indices[i]

        left_start = boundary_points[-1] if i == 0 else (peak_indices[i - 1] + peak) // 2
        left_region = profile[left_start:peak]
        if len(left_region) > 0:
            left_min_idx = left_start + int(np.argmin(left_region))
            boundary_points.append(left_min_idx)

    last_peak = peak_indices[-1]
    right_region = profile[last_peak:]
    if len(right_region) > 0:
        right_min_idx = last_peak + int(np.argmin(right_region))
        boundary_points.append(right_min_idx)
    boundary_points.append(n - 1)

    boundary_points = sorted(set(boundary_points))

    bp_x = np.array(boundary_points)
    bp_y = profile[bp_x]
    return np.interp(np.arange(n), bp_x, bp_y)


def compute_peak_areas(
    profile: NDArray[np.float64],
    peak_indices: NDArray[np.intp],
    baseline: NDArray[np.float64],  # noqa: ARG001 — kept for API compatibility, see docstring
    peak_properties: dict | None = None,
) -> list[float]:
    """Compute integrated intensity for each peak above baseline.

    The integrated intensity is the sum of the corrected signal within
    the peak region, proportional to protein amount in the band.

    NOTE: ``profile`` is expected to be the already baseline-corrected
    signal (i.e. ``corrected`` from ``analyze_lane``).  The ``baseline``
    argument is retained for API compatibility but is NOT subtracted again
    here — doing so would produce zero areas.

    Args:
        profile: 1-D baseline-corrected intensity profile (already ≥ 0).
        peak_indices: Indices of detected peaks.
        baseline: Estimated baseline (same length as profile) — kept for
            API compatibility, not used in the integration.
        peak_properties: Optional dict from ``find_peaks`` containing
            'left_ips' and 'right_ips' for peak width boundaries.

    Returns:
        List of integrated intensity values, one per peak.
    """
    # profile is already the corrected (baseline-subtracted) signal from
    # orient_profile_for_bands / analyze_lane.  Do NOT subtract baseline
    # again — that would zero everything out.
    corrected = np.maximum(profile, 0.0)
    areas = []

    for i, peak in enumerate(peak_indices):
        # Determine integration bounds from peak width
        if peak_properties and "left_ips" in peak_properties:
            half_w = peak_properties["right_ips"][i] - peak_properties["left_ips"][i]
            center = (peak_properties["left_ips"][i] + peak_properties["right_ips"][i]) / 2
            left = int(np.floor(center - half_w))
            right = int(np.ceil(center + half_w)) + 1
        else:
            half_window = 15
            left = max(0, peak - half_window)
            right = min(len(profile), peak + half_window)

        left = max(0, left)
        right = min(len(corrected), right)

        area = float(np.sum(corrected[left:right]))
        areas.append(area)

    return areas


def analyze_lane(
    image: NDArray[np.float64],
    lane_index: int,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    min_peak_height: float = 0.05,
    min_peak_distance: int = 10,
    min_snr: float = 3.0,
    max_band_width: int | None = None,
    min_band_width: int = 3,
    edge_margin_percent: float = 5.0,
    baseline_method: str = "rolling_ball",
    baseline_radius: int = 50,
    force_valleys_as_bands: bool | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[DetectedBand], bool]:
    """Full analysis pipeline for a single lane.

    Improved pipeline:
        1. Extract intensity profile.
        2. Estimate baseline and subtract it.
        3. Estimate noise level from the corrected signal.
        4. Detect peaks on the corrected signal with adaptive thresholds.
        5. Filter peaks by SNR, width, and prominence.
        6. Filter out peaks that are too close to the top/bottom edges.
        7. Compute integrated intensities.

    Args:
        image: Grayscale float64 image in [0.0, 1.0].
        lane_index: Zero-based index of this lane.
        x_start: Left column boundary.
        x_end: Right column boundary.
        y_start: Top row boundary.
        y_end: Bottom row boundary.
        min_peak_height: Minimum peak height threshold (absolute,
            applied to corrected signal).
        min_peak_distance: Minimum distance between peaks (pixels).
        min_snr: Minimum signal-to-noise ratio for a peak to be
            accepted. Higher = stricter, fewer false positives.
            Recommended: 2.0 (lenient) to 5.0 (strict).
        max_band_width: Maximum allowed band width in pixels. Bands
            wider than this are rejected. If None, defaults to 1/3
            of the lane height (no band should span > 1/3 of the gel).
        min_band_width: Minimum band width in pixels. Default 3.
        baseline_method: Either "rolling_ball" or "linear".
        baseline_radius: Radius for rolling ball baseline estimation.
        edge_margin_percent: Percentage (0-100) of the lane height at
            the top and bottom to ignore. Helps avoid false positives
            from rotation/cropping artifacts at the image edges. Default 5.

    Returns:
        Tuple of (display_profile, baseline, detected_bands, valleys_as_bands).
    """
    lane_height = y_end - y_start

    # Default max band width: 1/4 of lane height
    if max_band_width is None:
        max_band_width = max(10, lane_height // 4)

    # Step 1: Extract intensity profile (median is robust to dust/specks)
    profile = extract_lane_profile(image, x_start, x_end, y_start, y_end, statistic="median")

    # Keep the actual lane strip for 2D validation of peaks
    lane_width = x_end - x_start
    center = (x_start + x_end) // 2
    sample_half_width = max(1, int(lane_width * 0.5 / 2))
    strip_left = max(x_start, center - sample_half_width)
    strip_right = min(x_end, center + sample_half_width)
    lane_strip = image[y_start:y_end, strip_left:strip_right]

    # Step 2: Smooth profile lightly to reduce pixel noise
    smoothed = uniform_filter1d(profile, size=3)

    # Auto-compute baseline radius when 0 (Auto) is passed.
    if baseline_radius <= 0:
        baseline_radius = int(np.clip(lane_height * 0.40, 15, 300))

    # For dark-on-white blots, bands are valleys so we need a CEILING baseline.
    is_valleys = force_valleys_as_bands if force_valleys_as_bands is not None else True
    baseline_mode = "ceiling" if is_valleys else "floor"

    if baseline_method in {"rolling_ball", "linear"}:
        baseline = rolling_ball_baseline(smoothed, radius=baseline_radius, mode=baseline_mode)
    else:
        raise ValueError(
            f"Unknown baseline method '{baseline_method}'. Use 'rolling_ball' or 'linear'."
        )

    # Step 4: Orient the profile so that bands are positive peaks.
    corrected, display_profile, valleys_as_bands = orient_profile_for_bands(
        smoothed, baseline, force_valleys_as_bands=force_valleys_as_bands
    )

    # Step 5: Estimate noise from the corrected signal.
    slow = uniform_filter1d(corrected, size=31)
    residual = corrected - slow
    mad = float(np.median(np.abs(residual - np.median(residual))))
    noise = max(mad * 1.4826, 1e-6)

    # Step 6: Adaptive peak height threshold.
    adaptive_height = max(min_peak_height, noise * min_snr)

    # Adaptive prominence: must stand out from local surroundings
    adaptive_prominence = max(0.005, noise * 1.5)

    # Step 7: Detect peaks on the corrected signal
    peaks, properties = detect_peaks(
        corrected,
        min_peak_height=adaptive_height,
        min_peak_distance=min_peak_distance,
        min_prominence=adaptive_prominence,
        max_peak_width=float(max_band_width),
        min_peak_width=float(min_band_width),
    )

    # Step 7.5: Filter out peaks within the edge margins
    margin_px = int(lane_height * (max(0.0, float(edge_margin_percent)) / 100.0))
    valid_indices = []
    for i, p in enumerate(peaks):
        if margin_px <= p <= (lane_height - 1 - margin_px):
            valid_indices.append(i)

    if len(valid_indices) < len(peaks):
        peaks = peaks[valid_indices]
        for key in properties:
            properties[key] = properties[key][valid_indices]

    # Step 8: Refine baseline if using linear method
    if baseline_method == "linear" and len(peaks) > 0:
        baseline = linear_baseline(smoothed, peaks)
        corrected, display_profile, valleys_as_bands = orient_profile_for_bands(smoothed, baseline)

    # Step 9: Compute peak areas on the corrected signal.
    # Pass `corrected` as profile — compute_peak_areas expects an already
    # baseline-subtracted signal and will NOT subtract baseline again.
    areas = compute_peak_areas(corrected, peaks, baseline, properties)

    # If bands correspond to valleys in the original image, flip lane_strip
    lane_strip_for_score = 1.0 - lane_strip if valleys_as_bands else lane_strip

    # Step 10: Build band objects with SNR info
    bands = []
    for i, (peak, area) in enumerate(zip(peaks, areas, strict=False)):
        width = float(properties["widths"][i]) if "widths" in properties else 0.0
        peak_snr = float(corrected[peak] / noise) if noise > 0 else 0.0

        quality = _band_likeness_score(lane_strip_for_score, int(peak))
        min_quality = max(0.002, float(noise) * 0.8)
        if quality < min_quality:
            continue

        bands.append(
            DetectedBand(
                lane_index=lane_index,
                band_index=i,
                position=int(peak),
                peak_height=float(corrected[peak]),
                raw_height=float(smoothed[peak]),
                width=width,
                integrated_intensity=area,
                baseline_value=float(baseline[peak]),
                snr=round(peak_snr, 1),
                selected=True,
                quality=float(quality),
            )
        )

    return display_profile, baseline, bands, valleys_as_bands
