"""Western blot densitometry analysis pipeline.

This module provides the ``WesternBlotAnalyzer`` class — a high-level,
parameter-driven pipeline that automates the entire western blot
densitometry workflow:

    1. **Load** — Read any image format into a normalized array.
    2. **Preprocess** — Invert LUT if needed, rotate, crop.
    3. **Detect lanes** — Auto-detect or manually specify lane boundaries.
    4. **Detect bands** — Find peaks in each lane's intensity profile.
    5. **Compute densitometry** — Integrate band intensities above baseline.
    6. **Normalize** — Express densities relative to a reference lane.
    7. **Export** — Get results as a pandas DataFrame for analysis/export.

The pipeline is designed to be usable both programmatically (headless)
and via the Karcytics GUI. All parameters have sensible defaults but are
fully overridable.

Example::

    from karcytics_plugins.western_blot.analysis.western_blot import WesternBlotAnalyzer

    analyzer = WesternBlotAnalyzer()
    analyzer.load_image("my_blot.tif")
    analyzer.preprocess(invert_lut="auto")
    analyzer.detect_lanes(num_lanes=6)
    analyzer.detect_bands()
    analyzer.compute_densitometry()
    results = analyzer.get_results()
    results.to_csv("results.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from karcytics_sdk.contrib.image_utils import (
    adjust_contrast,
    auto_detect_inversion,
    crop_to_content,
    enhance_for_band_detection,
    invert_image,
    load_and_convert,
    rotate_image,
)
from karcytics_sdk.plugin import AnalysisBase
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

from .band_matching import (
    align_lanes_by_correlation,
    assign_matched_bands,
)
from .lane_detection import LaneROI, detect_lanes_projection
from .peak_analysis import (
    DetectedBand,
    analyze_lane,
    estimate_noise_level,
    extract_lane_profile,
    orient_profile_for_bands,
    rolling_ball_baseline,
)
from .state import AnalysisState  # noqa: F401 — re-exported for back-compat

logger = logging.getLogger(__name__)


class WesternBlotAnalyzer(AnalysisBase):
    """High-level western blot densitometry analyzer.

    This class orchestrates the entire analysis workflow, managing state
    and providing a clean API for both GUI and programmatic usage.

    Each analysis step modifies the internal state and can be re-run
    with different parameters without reloading the image.

    Attributes:
        state: ``AnalysisState`` object containing all intermediate results.

    Example::

        analyzer = WesternBlotAnalyzer()
        analyzer.load_image("blot.tif")
        analyzer.preprocess()
        analyzer.detect_lanes()
        analyzer.detect_bands()
        analyzer.compute_densitometry()
        df = analyzer.get_results()
    """

    def __init__(self, plugin_id: str = "western_blot") -> None:
        """Initialize an empty analyzer."""
        super().__init__(plugin_id)
        self.state = AnalysisState()
        # Transient per-task state, set by the caller (e.g. task_scheduler
        # call sites in ui/steps/*) immediately before run() is invoked.
        self.current_task_type: str = "auto"
        self.current_task_params: dict = {}

    def run(self, state: AnalysisState) -> dict:  # type: ignore[override]
        """Central entry point for background execution.

        Narrows the base class's ``state: PluginState | None`` — this is
        the SDK's own documented override pattern (see AnalysisBase.run's
        docstring example), not an accidental signature drift.

        The 'task_type' should be present in the state's results_df metadata
        or as a transient attribute on the analyzer.
        """
        task_type = self.current_task_type
        params = self.current_task_params

        if task_type == "detect_lanes":
            lanes = self.detect_lanes(**params)
            return {"lanes": lanes}

        if task_type == "detect_bands":
            bands = self.detect_bands(**params)
            # detect_bands also updates profiles, baselines, orientations
            return {
                "bands": bands,
                "profiles": self.state.profiles,
                "baselines": self.state.baselines,
                "lane_orientations": self.state.lane_orientations,
                "detection_image": self.state.detection_image,
            }

        if task_type == "compute_results":
            df = self.compute_densitometry(**params)
            return {"results_df": df}

        # "auto" or unknown
        if state.image_path is None:
            raise RuntimeError("run(): 'auto' task requires state.image_path to be set.")
        df = self.run_auto(state.image_path, **params)
        return {"results_df": df}

    # ------------------------------------------------------------------
    # Step 1: Load Image
    # ------------------------------------------------------------------

    def load_image(self, path: str | Path) -> NDArray[np.float64]:
        """Load a western blot image from disk.

        Reads the image and converts to grayscale float64 in [0, 1].
        Resets all downstream analysis state.

        Args:
            path: Path to the image file (TIFF, PNG, JPG, BMP, etc.).

        Returns:
            The loaded grayscale image.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read.
        """
        path = Path(path)
        logger.info("Loading image: %s", path)

        image = load_and_convert(path, as_grayscale=True)

        # Reset state — raw_image is the source of truth, never modified
        self.state = AnalysisState(
            image_path=path,
            raw_image=image.copy(),
            base_image=image.copy(),
            processed_image=image.copy(),
        )

        logger.info(
            "Loaded image: %dx%d pixels, dtype=%s",
            image.shape[1],
            image.shape[0],
            image.dtype,
        )
        return image

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        invert_lut: bool | Literal["auto"] = "auto",
        rotation_angle: float = 0.0,
        contrast_alpha: float = 1.0,
        contrast_beta: float = 0.0,
        manual_crop_rect: tuple[int, int, int, int] | None = None,
        auto_crop: bool = False,
    ) -> NDArray[np.float64]:
        """Preprocess the image for analysis.

        Applies LUT inversion, rotation, and optional auto-cropping.
        After preprocessing, the downstream peak-finding expects a
        standardized convention where **bands correspond to peaks**
        (higher intensity values) in the extracted lane profile.

        Args:
            invert_lut: Whether to invert the image LUT.
                - ``"auto"``: Auto-detect based on image characteristics.
                - ``True``: Force inversion.
                - ``False``: Skip inversion.
            rotation_angle: Angle in degrees to rotate the image.
                Positive = counter-clockwise. Use to straighten tilted gels.
            contrast_alpha: Contrast multiplier. Default 1.0.
            contrast_beta: Brightness offset. Default 0.0.
            manual_crop_rect: Tuple of (x, y, width, height) to crop the image after rotation.
            auto_crop: If True, crop whitespace borders.

        Returns:
            Preprocessed image.

        Raises:
            RuntimeError: If no image has been loaded.
        """
        self._require_image("preprocess")
        assert self.state.raw_image is not None  # guaranteed by _require_image

        # ── Stage 1: base_image — inversion + contrast + rotation ────────
        # Always rebuilt from raw_image so base_image is a stable coordinate
        # space.  Crop rects are always expressed in base_image coordinates.
        image = self.state.raw_image.copy()

        # LUT inversion
        if invert_lut == "auto":
            needs_inversion = auto_detect_inversion(image)
            logger.info("Auto-inversion detection: %s", needs_inversion)
        else:
            needs_inversion = bool(invert_lut)

        self.state.is_inverted = False
        if needs_inversion:
            image = invert_image(image)
            self.state.is_inverted = True
            logger.info("Applied LUT inversion")

        # Contrast/Brightness
        self.state.contrast_alpha = contrast_alpha
        self.state.contrast_beta = contrast_beta
        if abs(contrast_alpha - 1.0) > 0.001 or abs(contrast_beta) > 0.001:
            image = adjust_contrast(image, contrast_alpha, contrast_beta)
            logger.info("Applied contrast alpha=%.3f, beta=%.3f", contrast_alpha, contrast_beta)

        # Rotation
        self.state.rotation_angle = rotation_angle
        if abs(rotation_angle) > 0.01:
            image = rotate_image(image, rotation_angle)
            logger.info("Rotated image by %.2f degrees", rotation_angle)

        # Commit base_image — crop rects are always in these coordinates
        self.state.base_image = image.copy()

        # ── Stage 2: processed_image — crop applied to base_image ────────
        # manual_crop_rect coords are in base_image space — always correct
        # regardless of how many times the user crops.
        self.state.manual_crop_rect = manual_crop_rect
        if manual_crop_rect is not None:
            x, y, w, h = manual_crop_rect
            bh, bw = image.shape[:2]
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + h, bh)
            x_end = min(x + w, bw)
            if y_end > y_start and x_end > x_start:
                image = image[y_start:y_end, x_start:x_end]
                logger.info(
                    "Applied crop: %s → %dx%d", manual_crop_rect, image.shape[1], image.shape[0]
                )

        if auto_crop:
            image = crop_to_content(image)
            logger.info("Auto-cropped to content")

        self.state.processed_image = image

        # Clear downstream state — lanes/bands are invalidated by image change
        self.state.lanes = []
        self.state.profiles = []
        self.state.baselines = []
        self.state.bands = []
        self.state.results_df = None

        return image

    # ------------------------------------------------------------------
    # Step 3: Detect Lanes
    # ------------------------------------------------------------------

    def detect_lanes(
        self,
        num_lanes: int | None = None,
        smoothing_window: int = 15,
    ) -> list[LaneROI]:
        """Detect lane boundaries in the blot image.

        Uses vertical intensity projection to find inter-lane gaps.
        If ``num_lanes`` is provided, selects that many lanes from
        the detected boundaries.

        Args:
            num_lanes: Expected number of lanes. If None, auto-detect.
            smoothing_window: Smoothing kernel width for the projection
                profile. Increase for noisy images.

        Returns:
            List of ``LaneROI`` objects, ordered left-to-right.

        Raises:
            RuntimeError: If no image has been preprocessed.
        """
        self._require_image("detect_lanes")
        # processed_image is always set alongside raw_image in load_image(),
        # so _require_image (which checks raw_image) guarantees this too.
        assert self.state.processed_image is not None

        image = self.state.processed_image
        lanes = detect_lanes_projection(
            image,
            num_lanes=num_lanes,
            smoothing_window=smoothing_window,
        )

        self.state.lanes = lanes

        # Clear downstream state
        self.state.profiles = []
        self.state.baselines = []
        self.state.bands = []
        self.state.results_df = None

        logger.info("Detected %d lanes", len(lanes))
        return lanes

    # ------------------------------------------------------------------
    # Step 4: Detect Bands
    # ------------------------------------------------------------------

    def detect_bands(
        self,
        min_peak_height: float = 0.02,
        min_peak_distance: int = 10,
        min_snr: float = 3.0,
        max_band_width: int | None = None,
        min_band_width: int = 3,
        edge_margin_percent: float = 5.0,
        baseline_method: str = "rolling_ball",
        baseline_radius: int = 50,
        enhance_image: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        denoise_median_ksize: int = 0,
        background_kernel_size: int = 0,
        manual_pick: bool = False,
        force_valleys_as_bands: bool | None = None,
    ) -> list[DetectedBand]:
        """Detect bands in all lanes and extract intensity profiles.

        Uses adaptive noise-based thresholding: peaks must exceed both
        the absolute ``min_peak_height`` AND a threshold derived from
        the estimated noise level (``noise * min_snr``).

        Args:
            min_peak_height: Minimum peak height to count as a band.
                Range [0, 1]. Lower = more sensitive.
            min_peak_distance: Minimum distance between bands (pixels).
            min_snr: Minimum signal-to-noise ratio for a band.
                Higher values = stricter filtering. Recommended 2-5.
            max_band_width: Maximum allowed band width in pixels.
                Bands wider than this are rejected (they're likely
                background artifacts). If None, auto-sized to 1/4
                of the lane height.
            min_band_width: Minimum band width in pixels (default 3).
                Bands narrower than this are rejected (noise spikes).
            edge_margin_percent: Percentage of lane height at top and
                bottom to ignore. Helps avoid rotation/cropping edge artifacts.
            baseline_method: Baseline estimation method.
                - ``"rolling_ball"``: Rolling ball algorithm (default).
                - ``"linear"``: Linear interpolation between valleys.
            baseline_radius: Radius for the rolling ball baseline.
                Only used when ``baseline_method="rolling_ball"``.
            enhance_image: If True, apply image enhancement (CLAHE, denoising, background correction)
                before band detection. This can improve peak finding on noisy or uneven images.
            clahe_clip_limit: Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).
                Higher values increase contrast. Only used if ``enhance_image`` is True.
            clahe_tile_grid_size: Tile grid size for CLAHE. Only used if ``enhance_image`` is True.
            denoise_median_ksize: Kernel size for median denoising. 0 to disable.
                Only used if ``enhance_image`` is True.
            background_kernel_size: Kernel size for rolling ball background subtraction. 0 to disable.
                Only used if ``enhance_image`` is True.
            manual_pick: If True, only compute profiles and baselines,
                allowing for manual band picking later. No automatic band detection.
            force_valleys_as_bands: Manual override for profile orientation.

        Returns:
            List of all ``DetectedBand`` objects across all lanes.

        Raises:
            RuntimeError: If lanes have not been detected.
        """
        self._require_lanes("detect_bands")
        # processed_image is always set alongside raw_image in load_image();
        # _require_lanes calls _require_image (checks raw_image) first.
        assert self.state.processed_image is not None

        # Default to valleys-as-bands for dark-on-white western blots.
        # The user can override via force_valleys_as_bands=False if needed.
        if force_valleys_as_bands is None:
            force_valleys_as_bands = True

        image = self.state.processed_image

        # NOTE: enhance_image (CLAHE) is intentionally NOT applied by default.
        # CLAHE normalizes local contrast, which flattens the very intensity
        # differences (band vs background) that valley detection relies on.
        # Only apply if the caller explicitly requests it.
        if enhance_image:
            if background_kernel_size == 0:
                h, w = image.shape[:2]
                k = int(max(0, round(min(h, w) * 0.05)))
                k = max(0, min(k, 151))
                if k % 2 == 0:
                    k += 1
                background_kernel_size = k if k >= 21 else 0

            image = enhance_for_band_detection(
                image,
                apply_clahe=True,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                denoise_median_ksize=denoise_median_ksize,
                background_kernel_size=background_kernel_size,
            )

        # Store the image actually used for profiling so that detect_bands_for_lane
        # (called by the flip button) uses the exact same pixel data.
        self.state.detection_image = image
        all_bands = []
        profiles = []
        baselines = []
        orientations = []

        for lane in self.state.lanes:
            if manual_pick:
                # ImageJ-style workflow: compute profiles/baselines only.
                raw_profile = extract_lane_profile(
                    image,
                    lane.x_start,
                    lane.x_end,
                    lane.y_start,
                    lane.y_end,
                    statistic="median",
                )
                smoothed = uniform_filter1d(raw_profile, size=3)
                if baseline_method in {"rolling_ball", "linear"}:
                    baseline = rolling_ball_baseline(smoothed, radius=baseline_radius)
                else:
                    raise ValueError(
                        f"Unknown baseline method '{baseline_method}'. Use 'rolling_ball' or 'linear'."
                    )

                # Orient profile so that bands are peaks even if they appear
                # as valleys in the original image.
                corrected, oriented_profile, valleys_as_bands = orient_profile_for_bands(
                    smoothed, baseline, force_valleys_as_bands=force_valleys_as_bands
                )
                profile = oriented_profile
                orientations.append(valleys_as_bands)
                bands: list[DetectedBand] = []
            else:
                profile, baseline, bands, valleys_as_bands = analyze_lane(
                    image,
                    lane_index=lane.index,
                    x_start=lane.x_start,
                    x_end=lane.x_end,
                    y_start=lane.y_start,
                    y_end=lane.y_end,
                    min_peak_height=min_peak_height,
                    min_peak_distance=min_peak_distance,
                    min_snr=min_snr,
                    max_band_width=max_band_width,
                    min_band_width=min_band_width,
                    edge_margin_percent=edge_margin_percent,
                    baseline_method=baseline_method,
                    baseline_radius=baseline_radius,
                    force_valleys_as_bands=force_valleys_as_bands,
                )
                orientations.append(valleys_as_bands)
            profiles.append(profile)
            baselines.append(baseline)
            all_bands.extend(bands)

        self.state.bands = all_bands
        self.state.profiles = profiles
        self.state.baselines = baselines
        self.state.lane_orientations = orientations
        self.state.results_df = None

        logger.info(
            "Detected %d bands across %d lanes",
            len(all_bands),
            len(self.state.lanes),
        )
        return all_bands

    def add_manual_band(
        self,
        lane_index: int,
        y_position: int,
        *,
        search_window: int = 18,
        valley_window: int = 35,
        min_peak_snr: float = 2.5,  # noqa: ARG002 — unused, pre-existing
        auto_snap: bool = True,  # <--- NEW PARAMETER
    ) -> DetectedBand | None:
        """Add a band by clicking (ImageJ wand-style)."""
        self._require_lanes("add_manual_band")
        if not self.state.profiles or not self.state.baselines:
            raise RuntimeError("No lane profiles available. Run Detect Bands first.")

        lane_index = int(lane_index)
        if lane_index < 0 or lane_index >= len(self.state.lanes):
            return None

        display_profile = np.asarray(self.state.profiles[lane_index], dtype=np.float64)
        baseline = np.asarray(self.state.baselines[lane_index], dtype=np.float64)
        smoothed = uniform_filter1d(display_profile, size=3)
        corrected = np.maximum(smoothed - baseline, 0.0)

        noise = estimate_noise_level(display_profile, baseline)
        y0 = int(np.clip(int(y_position), 0, len(corrected) - 1))

        # --- NEW: THE AUTO-SNAP BYPASS LOGIC ---
        if auto_snap:
            left = max(0, y0 - int(search_window))
            right = min(len(corrected), y0 + int(search_window) + 1)
            if right - left < 3:
                return None

            peak = left + int(np.argmax(corrected[left:right]))

            # Relaxed SNR: only require 1.0x noise if snapping
            if corrected[peak] < max(1.0 * noise, 1e-4):
                return None
        else:
            # MANUAL OVERRIDE: Drop the peak EXACTLY where they clicked.
            # We skip the SNR check entirely because we trust the user's eyes over the math.
            peak = y0
        # ---------------------------------------

        # Find valleys to define the base line segment
        vleft = max(0, peak - int(valley_window))
        vright = min(len(corrected), peak + int(valley_window) + 1)
        if peak - vleft < 2 or vright - peak < 2:
            return None

        left_valley = vleft + int(np.argmin(corrected[vleft:peak]))
        right_valley = peak + int(np.argmin(corrected[peak:vright]))
        if right_valley <= left_valley:
            return None

        # Straight baseline connecting the smoothed display_profile at the valleys
        base_line = np.linspace(
            float(smoothed[left_valley]),
            float(smoothed[right_valley]),
            right_valley - left_valley + 1,
        )
        local = smoothed[left_valley : right_valley + 1]
        local_corrected = np.maximum(local - base_line, 0.0)
        area = float(np.sum(local_corrected))

        half = float(np.max(local_corrected)) * 0.5
        width = float(np.sum(local_corrected >= half)) if half > 0 else 0.0

        existing = [b for b in self.state.bands if b.lane_index == lane_index]
        band_idx = len(existing)

        band = DetectedBand(
            lane_index=lane_index,
            band_index=band_idx,
            position=int(peak),
            peak_height=float(corrected[peak]),
            raw_height=float(smoothed[peak]),
            width=width,
            integrated_intensity=area,
            baseline_value=float(baseline[peak]),
            snr=round(float(corrected[peak] / noise), 1) if noise > 0 else 10.0,
            selected=True,
        )

        self.state.bands.append(band)
        self.state.results_df = None
        return band

    def add_manual_band_range(
        self, lane_idx: int, y_start: float, y_end: float, auto_snap: bool = True
    ):
        """Creates a band exactly where the user dragged a bounding box."""
        import numpy as np

        from .peak_analysis import DetectedBand

        if not self.state.profiles or lane_idx >= len(self.state.profiles):
            return None

        y0 = int(round(min(y_start, y_end)))
        y1 = int(round(max(y_start, y_end)))

        profile = self.state.profiles[lane_idx]
        baseline = self.state.baselines[lane_idx]

        y0 = max(0, y0)
        y1 = min(len(profile) - 1, y1)
        if y1 <= y0:
            return None

        corrected = np.maximum(profile - baseline, 0)
        segment = corrected[y0 : y1 + 1]

        # --- NEW: THE AUTO-SNAP BYPASS LOGIC ---
        if auto_snap:
            # Find the mathematical highest point inside the dragged box
            local_peak_idx = int(np.argmax(segment))
            peak_y = y0 + local_peak_idx
        else:
            # MANUAL OVERRIDE: Put the marker exactly in the dead center of the box
            peak_y = y0 + (len(segment) // 2)
        # ---------------------------------------

        peak_height = float(corrected[peak_y])
        raw_height = float(profile[peak_y])
        area = float(np.sum(segment))

        band = DetectedBand(
            lane_index=lane_idx,
            band_index=len([b for b in self.state.bands if b.lane_index == lane_idx]),
            position=peak_y,
            peak_height=peak_height,
            raw_height=raw_height,
            width=float(y1 - y0),
            integrated_intensity=area,
            baseline_value=float(baseline[peak_y]),
            snr=10.0,
            selected=True,
        )

        self.state.bands.append(band)
        self.state.bands.sort(key=lambda b: (b.lane_index, b.position))

        return band

    def remove_band(self, lane_index: int, band_index: int) -> bool:
        """Remove a band from the specific lane by its index.

        Args:
            lane_index: The lane index.
            band_index: The band index within that lane.

        Returns:
            True if removed, False if not found.
        """
        for i, b in enumerate(self.state.bands):
            if b.lane_index == lane_index and b.band_index == band_index:
                self.state.bands.pop(i)
                self.state.results_df = None

                # Re-index remaining bands in this lane
                lane_bands = [band for band in self.state.bands if band.lane_index == lane_index]
                for j, band in enumerate(lane_bands):
                    band.band_index = j
                return True
        return False

    def remove_band_at(self, lane_index: int, position: float, tolerance: float = 10.0) -> bool:
        """Remove a band near a specific vertical position.

        Args:
            lane_index: The lane index.
            position: The y-position (profile coordinate).
            tolerance: Max distance in pixels to consider a match.

        Returns:
            True if removed, False if not found.
        """
        best_match = None
        min_dist = tolerance

        for i, b in enumerate(self.state.bands):
            if b.lane_index == lane_index:
                dist = abs(b.position - position)
                if dist < min_dist:
                    min_dist = dist
                    best_match = i

        if best_match is not None:
            self.state.bands.pop(best_match)
            self.state.results_df = None

            # Re-index
            lane_bands = [band for band in self.state.bands if band.lane_index == lane_index]
            for j, band in enumerate(lane_bands):
                band.band_index = j
            return True
        return False

    def detect_bands_for_lane(
        self,
        lane_index: int,
        force_valleys_as_bands: bool | None = None,
        **detection_params,
    ) -> list[DetectedBand]:
        """Re-run band detection for a single lane, optionally with orientation override.

        Useful for manually flipping orientation for just one problematic lane.
        """
        self._require_lanes("detect_bands_for_lane")
        # processed_image is always set alongside raw_image in load_image();
        # _require_lanes calls _require_image (checks raw_image) first.
        assert self.state.processed_image is not None
        if lane_index < 0 or lane_index >= len(self.state.lanes):
            raise ValueError(f"Invalid lane index {lane_index}")

        lane = self.state.lanes[lane_index]

        # Use the same enhanced image that detect_bands used, not the raw processed image.
        # This ensures the flip button produces a consistent profile.
        # NOTE: must be `is not None`, not `or` — `array or x` raises ValueError
        # ("truth value of an array is ambiguous") once detection_image is set.
        image = (
            self.state.detection_image
            if self.state.detection_image is not None
            else self.state.processed_image
        )

        # Default to valleys-as-bands (dark-on-white blot) unless overridden.
        if force_valleys_as_bands is None:
            force_valleys_as_bands = True

        # Merge defaults with provided params
        params = {
            "min_peak_height": 0.02,
            "min_peak_distance": 10,
            "min_snr": 3.0,
            "max_band_width": None,
            "min_band_width": 3,
            "edge_margin_percent": 5.0,
            "baseline_method": "rolling_ball",
            "baseline_radius": 0,  # 0 = auto (20% of lane height)
        }
        params.update(detection_params)

        # Analyze just this lane
        from .peak_analysis import analyze_lane

        # params merges fixed defaults with arbitrary caller overrides, so
        # its value type is a union across analyze_lane's several distinctly
        # typed keyword parameters — mypy can't verify a dict-splat against
        # that without a TypedDict, which would be overkill for one call site.
        profile, baseline, bands, auto_valleys = analyze_lane(
            image,
            lane_index=lane.index,
            x_start=lane.x_start,
            x_end=lane.x_end,
            y_start=lane.y_start,
            y_end=lane.y_end,
            force_valleys_as_bands=force_valleys_as_bands,
            **params,  # type: ignore[arg-type]
        )

        # Update orientation state
        if lane_index < len(self.state.lane_orientations):
            self.state.lane_orientations[lane_index] = auto_valleys
        else:
            # Should not happen but just in case
            self.state.lane_orientations.append(auto_valleys)

        # Update state for this lane
        if len(self.state.profiles) > lane_index:
            self.state.profiles[lane_index] = profile
        if len(self.state.baselines) > lane_index:
            self.state.baselines[lane_index] = baseline

        # Remove old bands for this lane and add new ones
        self.state.bands = [b for b in self.state.bands if b.lane_index != lane_index]
        self.state.bands.extend(bands)
        self.state.results_df = None

        return bands

    # ------------------------------------------------------------------
    # Step 5: Compute Densitometry (Normalization)
    # ------------------------------------------------------------------

    def compute_densitometry(
        self,
        reference_lane: int | None = None,
        control_band_index: int = 0,
        normalize_control_to_one: bool = True,
        lane_types: dict[int, str] | None = None,
        match_bands_across_lanes: bool = True,
        matching_tolerance_px: float = 12.0,
        alignment_max_shift_px: int = 40,
        ponceau_loading_factors: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Compute normalized densitometry values.

        Builds a results DataFrame with raw and normalized intensities.

        Normalization:
            1. If ``reference_lane`` is specified, each band's intensity
               is divided by the corresponding band's intensity in the
               reference lane (e.g., Ponceau S loading control).
            2. If ``normalize_control_to_one`` is True, the first lane's
               (or control lane's) ratio is set to 1.0 and all others
               are expressed relative to it.

        Args:
            reference_lane: Lane index to use as reference/denominator.
                If None, normalization is done as percentage of total.
            control_band_index: Which band index within the reference
                lane to use for normalization. Default 0 (first/primary).
            normalize_control_to_one: If True, scale results so that
                the first lane's normalized value equals 1.0.

        Returns:
            pandas DataFrame with columns:
                - lane: Lane index.
                - band: Band index within the lane.
                - position: Pixel position in the profile.
                - raw_intensity: Integrated intensity above baseline.
                - percent_of_total: Intensity as % of all bands.
                - normalized: Normalized intensity value.

        Args:
            ponceau_loading_factors: Optional dict mapping WB lane index to
                Ponceau loading factor (centred on 1.0). When provided,
                ``ponceau_normalized = normalized / factor`` is added.

        Raises:
            RuntimeError: If bands have not been detected.
        """
        self._require_bands("compute_densitometry")

        lane_types = lane_types or {}

        # Filter out deactivated bands and excluded lanes
        active_bands = []
        for b in self.state.bands:
            if not getattr(b, "selected", True):
                continue
            l_type = lane_types.get(b.lane_index, "Sample")
            if l_type == "Exclude":
                continue
            active_bands.append(b)

        # Optional: align lanes and match corresponding bands across lanes.
        if match_bands_across_lanes and active_bands and self.state.profiles:
            # Align only sample lanes (ladders/excluded lanes are not used for alignment).
            sample_lanes = sorted(
                {
                    b.lane_index
                    for b in active_bands
                    if lane_types.get(b.lane_index, "Sample") == "Sample"
                }
            )
            if sample_lanes:
                ref_lane = sample_lanes[0]
                alignments = align_lanes_by_correlation(
                    self.state.profiles,
                    self.state.baselines,
                    ref_lane=ref_lane,
                    max_shift_px=alignment_max_shift_px,
                    lane_indices=sample_lanes,
                )
                lane_to_shift = {k: v.shift_px for k, v in alignments.items()}

                lane_to_positions: dict[int, list[tuple[int, float]]] = {}
                for b in active_bands:
                    if lane_types.get(b.lane_index, "Sample") != "Sample":
                        continue
                    lane_to_positions.setdefault(b.lane_index, []).append(
                        (b.band_index, float(b.position))
                    )

                matched = assign_matched_bands(
                    lane_to_band_positions=lane_to_positions,
                    lane_to_shift=lane_to_shift,
                    tolerance_px=matching_tolerance_px,
                )

                for b in active_bands:
                    if lane_types.get(b.lane_index, "Sample") != "Sample":
                        continue
                    key = (b.lane_index, b.band_index)
                    if key in matched:
                        mb, aligned_pos = matched[key]
                        b.matched_band = int(mb)
                        b.aligned_position = float(aligned_pos)
                        b.match_score = float(
                            abs(
                                float(b.position)
                                + float(lane_to_shift.get(b.lane_index, 0))
                                - aligned_pos
                            )
                        )

        # Recalculate total intensity only for samples (ladders shouldn't skew the total)
        sample_intensity = sum(
            b.integrated_intensity
            for b in active_bands
            if lane_types.get(b.lane_index, "Sample") == "Sample"
        )

        records = []
        for band in active_bands:
            is_ladder = lane_types.get(band.lane_index, "Sample") == "Ladder"

            if is_ladder:
                pct = 0.0
            else:
                pct = (
                    (band.integrated_intensity / sample_intensity * 100)
                    if sample_intensity > 0
                    else 0.0
                )

            records.append(
                {
                    "lane": band.lane_index,
                    "band": band.band_index,
                    "matched_band": getattr(band, "matched_band", None),
                    "position": band.position,
                    "aligned_position": getattr(band, "aligned_position", None),
                    "match_score": getattr(band, "match_score", None),
                    "peak_height": band.peak_height,
                    "width": round(band.width, 1),
                    "raw_intensity": band.integrated_intensity,
                    "baseline": band.baseline_value,
                    "snr": band.snr,
                    "percent_of_total": round(pct, 2),
                    "is_ladder": is_ladder,
                }
            )

        df = pd.DataFrame(records)

        # Normalization
        if reference_lane is not None and len(df) > 0:
            # Get reference lane band intensities
            ref_bands = df[(df["lane"] == reference_lane) & (df["band"] == control_band_index)]
            if len(ref_bands) > 0:
                ref_intensity = ref_bands.iloc[0]["raw_intensity"]
                if ref_intensity > 0:
                    df["normalized"] = df["raw_intensity"] / ref_intensity
                else:
                    df["normalized"] = 0.0
            else:
                df["normalized"] = df["raw_intensity"]
        elif len(df) > 0:
            # Normalize each band as fraction of total
            if sample_intensity > 0:
                df["normalized"] = df["raw_intensity"] / sample_intensity
            else:
                df["normalized"] = 0.0
        else:
            df["normalized"] = pd.Series(dtype=float)

        # Optionally set control to 1.0
        if normalize_control_to_one and len(df) > 0:
            first_lane_value = df.loc[df["lane"] == df["lane"].min(), "normalized"]
            if len(first_lane_value) > 0 and first_lane_value.iloc[0] > 0:
                scale_factor = 1.0 / first_lane_value.iloc[0]
                df["normalized"] = df["normalized"] * scale_factor

        # ── Ponceau loading correction ────────────────────────────────────
        if ponceau_loading_factors and len(df) > 0:
            df["ponceau_factor"] = df["lane"].map(
                lambda lane_idx: ponceau_loading_factors.get(int(lane_idx), 1.0)
            )
            df["ponceau_normalized"] = df.apply(
                lambda row: (
                    row["normalized"] / row["ponceau_factor"]
                    if row["ponceau_factor"] > 0
                    else row["normalized"]
                ),
                axis=1,
            )
        elif len(df) > 0:
            df["ponceau_factor"] = 1.0
            df["ponceau_normalized"] = df["normalized"]

        self.state.results_df = df
        logger.info("Computed densitometry for %d bands", len(df))
        return df

    # ------------------------------------------------------------------
    # Step 6: Get Results
    # ------------------------------------------------------------------

    def get_results(self) -> pd.DataFrame:
        """Get the analysis results as a pandas DataFrame.

        If ``compute_densitometry`` hasn't been called yet, runs it
        with default parameters.

        Returns:
            DataFrame with densitometry results.
        """
        if self.state.results_df is None:
            return self.compute_densitometry()
        return self.state.results_df.copy()

    def export_csv(self, path: str | Path) -> Path:
        """Export results to a CSV file.

        Args:
            path: Output file path.

        Returns:
            Path to the saved CSV file.
        """
        path = Path(path)
        df = self.get_results()
        df.to_csv(path, index=False)
        logger.info("Exported CSV to: %s", path)
        return path

    def export_excel(self, path: str | Path) -> Path:
        """Export results to an Excel file.

        The Excel file includes a formatted "Results" sheet with
        the densitometry data.

        Args:
            path: Output file path (should end in .xlsx).

        Returns:
            Path to the saved Excel file.
        """
        path = Path(path)
        df = self.get_results()
        df.to_excel(path, index=False, sheet_name="Densitometry Results")
        logger.info("Exported Excel to: %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience: Full Auto Pipeline
    # ------------------------------------------------------------------

    def run_auto(
        self,
        path: str | Path,
        num_lanes: int | None = None,
        min_peak_height: float = 0.05,
        reference_lane: int | None = None,
    ) -> pd.DataFrame:
        """Run the complete analysis pipeline with default parameters.

        Convenience method that runs all steps in sequence. For more
        control, call individual methods.

        Args:
            path: Path to the image file.
            num_lanes: Expected number of lanes (None = auto-detect).
            min_peak_height: Band detection sensitivity.
            reference_lane: Reference lane for normalization.

        Returns:
            DataFrame with final results.

        Example::

            analyzer = WesternBlotAnalyzer()
            results = analyzer.run_auto("blot.tif", num_lanes=6)
        """
        self.load_image(path)
        self.preprocess(invert_lut="auto")
        self.detect_lanes(num_lanes=num_lanes)
        self.detect_bands(min_peak_height=min_peak_height)
        return self.compute_densitometry(reference_lane=reference_lane)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_image(self, step_name: str) -> None:
        """Ensure an image has been loaded."""
        if self.state.raw_image is None:
            raise RuntimeError(
                f"Cannot run '{step_name}': no image loaded. Call load_image() first."
            )

    def _require_lanes(self, step_name: str) -> None:
        """Ensure lanes have been detected."""
        self._require_image(step_name)
        if not self.state.lanes:
            raise RuntimeError(
                f"Cannot run '{step_name}': no lanes detected. Call detect_lanes() first."
            )

    def _require_bands(self, step_name: str) -> None:
        """Ensure bands have been detected."""
        self._require_lanes(step_name)
        if not self.state.bands:
            raise RuntimeError(
                f"Cannot run '{step_name}': no bands detected. Call detect_bands() first."
            )
