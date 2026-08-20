"""Analysis pipeline state container.

``AnalysisState`` is a plain dataclass that holds every intermediate
result produced by an analysis pipeline run.  It is intentionally kept
separate from the analyzer classes so that:

- Multiple analyzer types (WesternBlot, Ponceau, SDS-PAGE, …) can each
  own their own state instance without any cross-dependency.
- UI code, dialogs, and tests can import the type without pulling in the
  full analyzer.
- The shape of the data contract is visible in one place.

Nothing in this module performs computation — it is a data container only.

Image layer model
-----------------
Three image fields form a deliberate pipeline:

``raw_image``
    Pixels loaded directly from disk.  Never modified after load.
    Source of truth for all re-processing.

``base_image``
    ``raw_image`` after inversion, contrast adjustment, and rotation —
    but **before** any crop.  This is the stable coordinate space: crop
    rectangles are always expressed in ``base_image`` pixel coordinates.
    Recomputed whenever inversion/contrast/rotation changes.

``processed_image``  (= analysis image)
    ``base_image`` after the crop rectangle is applied.  This is what
    lane detection and band detection actually operate on.  When no crop
    is active, ``processed_image is base_image``.

``detection_image``
    Set only when optional CLAHE/denoising enhancement is applied inside
    ``detect_bands``.  Band detection uses this instead of
    ``processed_image`` so that profile-based operations (flip, manual
    add) always see the same pixels as the initial detection pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from karcytics_sdk.plugin import PluginState
from numpy.typing import NDArray

from .lane_detection import LaneROI
from .peak_analysis import DetectedBand


@dataclass
class AnalysisState(PluginState):
    """Mutable state for a single analysis run.

    Each field maps to one pipeline stage.  Fields are reset by the
    analyzer whenever an upstream step is re-run, so the state is always
    internally consistent.

    Attributes:
        image_path:       Path to the source image file on disk.
        raw_image:        Raw loaded image — never modified after load.
                          Alias ``original_image`` is preserved for
                          back-compatibility.
        base_image:       Image after inversion, rotation, and contrast —
                          but before crop.  Crop rects are always in
                          these pixel coordinates.
        processed_image:  Image after crop applied to ``base_image``.
                          This is what lane/band detection operates on.
                          Equal to ``base_image`` when no crop is active.
        detection_image:  Image actually used for band detection profiles
                          (may differ from ``processed_image`` if optional
                          CLAHE enhancement was applied).
        is_inverted:      Whether LUT inversion was applied.
        rotation_angle:   Rotation applied in degrees (positive = CCW).
        contrast_alpha:   Contrast multiplier (output = α·pixel + β).
        contrast_beta:    Brightness offset.
        manual_crop_rect: Active crop as (x, y, width, height) in
                          ``base_image`` pixel coordinates, or None.
        lanes:            Detected lane boundaries, ordered left-to-right.
        profiles:         Per-lane 1-D intensity profiles.
        baselines:        Per-lane estimated background baselines.
        lane_orientations: Per-lane flag — True when bands were originally
                           valleys and were flipped to peaks.
        bands:            All detected bands across all lanes.
        results_df:       Final densitometry DataFrame, or None.
    """

    # ── Image layers ──────────────────────────────────────────────────
    image_path: Path | None = None

    # Raw pixels from disk — never touched after load
    raw_image: NDArray[np.float64] | None = None

    # Post inversion+contrast+rotation, pre-crop — stable coord space
    base_image: NDArray[np.float64] | None = None

    # Post crop (= base_image when no crop) — used by analysis pipeline
    processed_image: NDArray[np.float64] | None = None

    # Optional enhanced image used by detect_bands
    detection_image: NDArray[np.float64] | None = None

    # ── Back-compat alias ─────────────────────────────────────────────
    # Code that references state.original_image still works.
    @property
    def original_image(self) -> NDArray[np.float64] | None:
        return self.raw_image

    @original_image.setter
    def original_image(self, value: NDArray[np.float64] | None) -> None:
        self.raw_image = value

    # ── Preprocessing parameters ──────────────────────────────────────
    is_inverted: bool = False
    rotation_angle: float = 0.0
    contrast_alpha: float = 1.0
    contrast_beta: float = 0.0

    # Crop rect in base_image coordinates
    manual_crop_rect: tuple[int, int, int, int] | None = None

    # ── Lane detection ─────────────────────────────────────────────────
    lanes: list[LaneROI] = field(default_factory=list)

    # ── Band detection ─────────────────────────────────────────────────
    profiles: list[NDArray[np.float64]] = field(default_factory=list)
    baselines: list[NDArray[np.float64]] = field(default_factory=list)
    lane_orientations: list[bool] = field(default_factory=list)
    bands: list[DetectedBand] = field(default_factory=list)

    # ── Densitometry ──────────────────────────────────────────────────
    results_df: pd.DataFrame | None = None

    # ── UI/wizard position, attached by WesternBlotPanel.get_state() ───
    metadata: dict = field(default_factory=dict)

    # Inside AnalysisState (or your module's main analyzer)

    def to_workflow_dict(self) -> dict:
        """Serializes the complete, exact biological state and math."""
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "preprocessing": {
                "is_inverted": getattr(self, "is_inverted", False),
                "rotation_angle": getattr(self, "rotation_angle", 0.0),
                "contrast_alpha": getattr(self, "contrast_alpha", 1.5),  # <-- Added
                "contrast_beta": getattr(self, "contrast_beta", -0.7),  # <-- Added
                "manual_crop_rect": getattr(self, "manual_crop_rect", None),
            },
            "lanes": [
                {
                    "index": getattr(lane, "index", 0),
                    "x_start": getattr(lane, "x_start", 0),
                    "x_end": getattr(lane, "x_end", 0),
                    "y_start": getattr(lane, "y_start", 0),
                    "y_end": getattr(lane, "y_end", 0),
                }
                for lane in self.lanes
            ],
            "bands": [
                {
                    "lane_index": getattr(b, "lane_index", 0),
                    "band_index": getattr(b, "band_index", 0),
                    "position": getattr(b, "position", 0),
                    "width": getattr(b, "width", 0),
                    "peak_height": getattr(b, "peak_height", 0.0),
                    "raw_height": getattr(b, "raw_height", 0.0),
                    "integrated_intensity": getattr(b, "integrated_intensity", 0.0),
                    "snr": getattr(b, "snr", 0.0),
                    "matched_band": getattr(b, "matched_band", None),
                }
                for b in self.bands
            ],
        }

    def from_workflow_dict(self, data: dict) -> None:
        """Restores the exact biological state without recalculation."""
        from .lane_detection import LaneROI
        from .peak_analysis import DetectedBand

        pre = data.get("preprocessing", {})
        self.is_inverted = pre.get("is_inverted", False)
        self.rotation_angle = pre.get("rotation_angle", 0.0)
        self.contrast_alpha = pre.get("contrast_alpha", 1.5)
        self.contrast_beta = pre.get("contrast_beta", -0.7)
        self.manual_crop_rect = pre.get("manual_crop_rect")

        self.lanes = []
        for l_data in data.get("lanes", []):
            # Using explicit kwargs to bypass strict __init__ positional requirements
            lane = LaneROI(
                index=l_data.get("index", 0),
                x_start=l_data.get("x_start", 0),
                x_end=l_data.get("x_end", 100),
                y_start=l_data.get("y_start", 0),
                y_end=l_data.get("y_end", 9999),
            )
            self.lanes.append(lane)

        self.bands = []
        for b_data in data.get("bands", []):
            band = DetectedBand(
                lane_index=b_data.get("lane_index", 0),
                band_index=b_data.get("band_index", 0),
                position=b_data.get("position", 0),
                width=b_data.get("width", 0),
                peak_height=b_data.get("peak_height", 0.0),
                raw_height=b_data.get("raw_height", 0.0),
                integrated_intensity=b_data.get("integrated_intensity", 0.0),
                matched_band=b_data.get("matched_band"),
            )
            band.snr = b_data.get("snr", 0.0)
            self.bands.append(band)
