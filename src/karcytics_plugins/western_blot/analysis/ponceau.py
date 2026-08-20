"""Ponceau S stain analysis for western blot loading normalization.

Ponceau S is a reversible total-protein stain applied to the membrane
before antibody probing.  Because it stains all proteins proportional
to their mass, the total Ponceau signal per lane reflects how much
protein was actually loaded — correcting for pipetting errors and
transfer inefficiencies.

Pipeline
--------
The Ponceau image is processed with the same steps as a western blot
(load → preprocess → detect lanes → detect bands), then:

1. For each lane, compute the **total integrated intensity** of all
   detected bands (or selected bands, depending on mode).
2. Express each lane's intensity as a fraction of the mean across all
   lanes — giving a **loading factor** centred on 1.0.
3. The WB densitometry step divides each lane's WB intensity by its
   Ponceau loading factor to produce the normalized result.

Normalization modes
-------------------
``"total"``  (default, recommended)
    Sum all detected bands in the lane.  Statistically robust because
    it averages over many protein species.

``"reference_band"``
    User picks one prominent band per lane (mirrors the ImageJ course
    protocol).  Less robust but easier to understand and audit.

Lane mapping
------------
The Ponceau image may have more or fewer lanes than the WB image (e.g.
extra ladder lanes, or failed lanes).  The user provides an explicit
mapping ``{ponceau_lane_idx: wb_lane_idx}`` so factors are applied to
the correct WB lane.  Unmapped WB lanes receive a factor of 1.0
(no correction applied).

Example::

    from karcytics_plugins.western_blot.analysis.ponceau import PonceauAnalyzer

    pon = PonceauAnalyzer()
    pon.load_image("ponceau.jpg")
    pon.preprocess(invert_lut="auto", contrast_alpha=2.0)
    pon.detect_lanes(num_lanes=6)
    pon.detect_bands(min_snr=2.0)          # lower SNR — faint pink bands
    factors = pon.get_loading_factors()    # {0: 0.94, 1: 1.12, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from karcytics_sdk.plugin import AnalysisBase

from .state import AnalysisState
from .western_blot import WesternBlotAnalyzer

logger = logging.getLogger(__name__)


def _band_intensity(band) -> float:
    """Return the best available intensity for a band.

    Prefers ``integrated_intensity`` (area under the peak) but falls back
    to ``peak_height`` if it is zero or near-zero.  This handles the
    transition period while the compute_peak_areas fix propagates.
    """
    v = float(band.integrated_intensity)
    if v > 1e-6:
        return v
    return float(band.peak_height)


class PonceauAnalyzer(AnalysisBase):
    """Thin wrapper around WesternBlotAnalyzer for Ponceau S quantification.

    Delegates background tasks to the inner WesternBlotAnalyzer.
    """

    def __init__(self, plugin_id: str = "western_blot") -> None:
        super().__init__(plugin_id)
        # Delegate all image processing to WesternBlotAnalyzer
        self._wb = WesternBlotAnalyzer(plugin_id)
        self.lane_mapping: dict[int, int] = {}  # ponceau_idx → wb_idx
        self.mode: str = "reference_band"  # default matches prof protocol
        self.ref_band_indices: dict[int, int] = {}  # ponceau_lane_idx → band_idx
        # Transient per-task state, set by the caller (e.g. task_scheduler
        # call sites in ui/steps/*) immediately before run() is invoked.
        self.current_task_type: str = "auto"
        self.current_task_params: dict = {}

    def run(self, state: AnalysisState) -> dict:  # type: ignore[override]
        """Delegate background execution to the WB analyzer.

        Transfers transient task attributes before running. Narrows the base
        class's ``state: PluginState | None`` — see WesternBlotAnalyzer.run
        for why this mirrors the SDK's own documented override pattern.
        """
        self._wb.current_task_type = self.current_task_type
        self._wb.current_task_params = self.current_task_params
        return self._wb.run(state)

    # ── Expose WB analyzer interface transparently ────────────────────

    @property
    def state(self) -> AnalysisState:
        return self._wb.state

    def load_image(self, path: str | Path) -> np.ndarray:
        return self._wb.load_image(path)

    def preprocess(self, **kwargs) -> np.ndarray:
        return self._wb.preprocess(**kwargs)

    def detect_lanes(self, **kwargs):
        return self._wb.detect_lanes(**kwargs)

    def detect_bands(self, **kwargs):
        # Ponceau bands are faint — lower default SNR than WB
        kwargs.setdefault("min_snr", 2.0)
        kwargs.setdefault("min_peak_height", 0.01)
        kwargs.setdefault("force_valleys_as_bands", True)
        return self._wb.detect_bands(**kwargs)

    def detect_bands_for_lane(self, *args, **kwargs):
        return self._wb.detect_bands_for_lane(*args, **kwargs)

    def add_manual_band(self, lane_idx: int, y_pos: float, auto_snap: bool = True):
        import numpy as np

        from .peak_analysis import DetectedBand

        if not self.state.profiles or lane_idx >= len(self.state.profiles):
            return None

        profile = self.state.profiles[lane_idx]
        baseline = self.state.baselines[lane_idx]
        corrected = np.maximum(profile - baseline, 0)

        target_y = max(0, min(len(profile) - 1, int(round(y_pos))))
        peak_y = target_y

        # 1. BULLETPROOF AUTO-SNAP
        if auto_snap:
            window = 10
            y0 = max(0, target_y - window)
            y1 = min(len(profile) - 1, target_y + window)
            segment = corrected[y0 : y1 + 1]

            # Only snap if there is actually a positive peak nearby
            if len(segment) > 0 and np.max(segment) > 0:
                peak_y = y0 + int(np.argmax(segment))
            else:
                peak_y = target_y  # Fallback to exactly where you clicked

        # 2. PURGE EXACT DUPLICATES ONLY
        # Shrunk from 15px to 3px so you can place bands right next to each other!
        self.state.bands = [
            b
            for b in self.state.bands
            if not (b.lane_index == lane_idx and abs(b.position - peak_y) <= 3)
        ]
        # 3. FORCE MINIMUM VISIBILITY (The Zero-Math Fix)
        calculated_height = float(corrected[peak_y])
        visual_height = max(0.05, calculated_height)  # Guarantee it is never 0.0

        band = DetectedBand(
            lane_index=lane_idx,
            band_index=0,
            position=peak_y,
            peak_height=visual_height,
            raw_height=float(profile[peak_y]),
            width=5.0,
            integrated_intensity=max(0.1, visual_height * 5.0),  # Guarantee non-zero area
            baseline_value=float(baseline[peak_y]),
            snr=10.0,  # High confidence because it's manual
            selected=True,
        )

        self.state.bands.append(band)
        self.state.bands.sort(key=lambda b: (b.lane_index, b.position))

        current_lane = -1
        b_idx = 0
        for b in self.state.bands:
            if b.lane_index != current_lane:
                current_lane = b.lane_index
                b_idx = 0
            b.band_index = b_idx
            b_idx += 1

        return band

    def add_manual_band_range(
        self, lane_idx: int, y_start: float, y_end: float, auto_snap: bool = True
    ):
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

        # 1. Enforce minimum drag width so it doesn't abort
        if y1 == y0:
            y1 = min(len(profile) - 1, y0 + 1)
        if y1 < y0:
            return None

        corrected = np.maximum(profile - baseline, 0)
        segment = corrected[y0 : y1 + 1]

        if auto_snap and len(segment) > 0 and np.max(segment) > 0:
            local_peak_idx = int(np.argmax(segment))
            peak_y = y0 + local_peak_idx
        else:
            peak_y = int(round((y0 + y1) / 2.0))

        # 2. PURGE NEIGHBORS under the dragged box
        self.state.bands = [
            b
            for b in self.state.bands
            if not (b.lane_index == lane_idx and abs(b.position - peak_y) <= 3)
        ]

        # 3. FORCE MINIMUM VISIBILITY (The Zero-Math Fix)
        calculated_height = float(corrected[peak_y]) if len(segment) > 0 else 0.0
        visual_height = max(0.05, calculated_height)  # Guarantee it is never 0.0

        band = DetectedBand(
            lane_index=lane_idx,
            band_index=0,
            position=peak_y,
            peak_height=visual_height,
            raw_height=float(profile[peak_y]),
            # Guarantee a minimum visual width of 5px
            width=float(y1 - y0) if (y1 - y0) > 2 else 5.0,
            integrated_intensity=max(0.1, float(np.sum(segment))),  # Guarantee non-zero area
            baseline_value=float(baseline[peak_y]),
            snr=10.0,
            selected=True,
        )

        self.state.bands.append(band)
        self.state.bands.sort(key=lambda b: (b.lane_index, b.position))

        current_lane = -1
        b_idx = 0
        for b in self.state.bands:
            if b.lane_index != current_lane:
                current_lane = b.lane_index
                b_idx = 0
            b.band_index = b_idx
            b_idx += 1

        return band

    def remove_band_at(self, *args, **kwargs):
        return self._wb.remove_band_at(*args, **kwargs)

    # ── Loading factor computation ─────────────────────────────────────

    def get_loading_factors(
        self,
        lane_types: dict[int, str] | None = None,
    ) -> dict[int, float]:
        """Compute per-lane Ponceau loading factors (in Ponceau lane space).

        Returns a dict keyed by **Ponceau** lane index.  Use
        ``get_wb_loading_factors`` to get them keyed by WB lane index.

        Args:
            lane_types: Optional mapping of lane index to type string
                (``"Sample"`` / ``"Ladder"`` / ``"Exclude"``).

        Returns:
            Dict ``{ponceau_lane_idx: loading_factor}`` where factors are
            centred on 1.0 (mean of sample lanes = 1.0).
        """
        if not self.state.bands:
            return {}

        lane_types = lane_types or {}
        active_lanes = sorted(
            {
                b.lane_index
                for b in self.state.bands
                if getattr(b, "selected", True)
                # Accept all mapped lanes, explicitly reject ignore types
                and lane_types.get(b.lane_index, "Sample")
                not in ["Ladder", "Exclude", "None", "Unmapped"]
            }
        )

        if not active_lanes:
            return {}

        # Compute raw intensity per lane
        raw: dict[int, float] = {}
        for lane_idx in active_lanes:
            lane_bands = [
                b
                for b in self.state.bands
                if b.lane_index == lane_idx and getattr(b, "selected", True)
            ]
            if not lane_bands:
                raw[lane_idx] = 0.0
                logger.warning(
                    "Lane %d: no Ponceau bands detected — loading factor will be 1.0 "
                    "(no correction applied for this lane).",
                    lane_idx,
                )
                continue

            if self.mode == "total":
                raw[lane_idx] = sum(_band_intensity(b) for b in lane_bands)
            else:
                # reference_band mode
                ref_idx = self.ref_band_indices.get(lane_idx)
                if ref_idx is None:
                    raw[lane_idx] = sum(_band_intensity(b) for b in lane_bands)
                    logger.warning(
                        "Lane %d: no reference band selected — using total lane intensity.",
                        lane_idx,
                    )
                else:
                    ref_bands = [b for b in lane_bands if b.band_index == ref_idx]
                    if ref_bands:
                        raw[lane_idx] = _band_intensity(ref_bands[0])
                    else:
                        raw[lane_idx] = sum(_band_intensity(b) for b in lane_bands)
                        logger.warning(
                            "Lane %d: reference band index %d not found — "
                            "using total lane intensity.",
                            lane_idx,
                            ref_idx,
                        )

        # Normalise so mean = 1.0
        values = [v for v in raw.values() if v > 0]
        if not values:
            return dict.fromkeys(active_lanes, 1.0)

        mean_intensity = float(np.mean(values))
        if mean_intensity == 0:
            return dict.fromkeys(active_lanes, 1.0)

        factors = {
            lane_idx: (raw[lane_idx] / mean_intensity if raw[lane_idx] > 0 else 1.0)
            for lane_idx in active_lanes
        }

        logger.info(
            "Ponceau loading factors (mode=%s): %s",
            self.mode,
            {k: round(v, 3) for k, v in factors.items()},
        )
        return factors

    def get_wb_loading_factors(
        self,
        num_wb_lanes: int,
        lane_types: dict[int, str] | None = None,
    ) -> dict[int, float]:
        """Return loading factors keyed by **WB** lane index.

        Unmapped WB lanes get a factor of 1.0 (no correction).

        Args:
            num_wb_lanes: Total number of WB lanes.
            lane_types: Lane type mapping for Ponceau lanes.

        Returns:
            Dict ``{wb_lane_idx: loading_factor}`` for all WB lanes.
        """
        ponceau_factors = self.get_loading_factors(lane_types=lane_types)

        wb_factors: dict[int, float] = dict.fromkeys(range(num_wb_lanes), 1.0)

        for pon_idx, wb_idx in self.lane_mapping.items():
            if pon_idx in ponceau_factors and 0 <= wb_idx < num_wb_lanes:
                wb_factors[wb_idx] = ponceau_factors[pon_idx]

        return wb_factors

    def get_ponceau_raw_per_wb_lane(
        self,
        num_wb_lanes: int,
    ) -> dict[int, float]:
        """Return the raw Ponceau reference band intensity for each WB lane.

        This is the actual intensity of the selected Ponceau reference band
        used as the denominator in::

            ratio = WB_band_intensity / Ponceau_ref_band_intensity

        Uses ``integrated_intensity`` when available, falling back to
        ``peak_height`` for robustness.

        Returns:
            Dict mapping ``{wb_lane_idx: ponceau_raw_intensity}``.
            Lanes without a Ponceau mapping return 0.0.
        """
        result: dict[int, float] = dict.fromkeys(range(num_wb_lanes), 0.0)

        for pon_idx, wb_idx in self.lane_mapping.items():
            if wb_idx >= num_wb_lanes:
                continue
            lane_bands = [b for b in self.state.bands if b.lane_index == pon_idx]
            if not lane_bands:
                continue

            if self.mode == "reference_band":
                ref_idx = self.ref_band_indices.get(pon_idx)
                if ref_idx is not None:
                    ref_bands = [b for b in lane_bands if b.band_index == ref_idx]
                    if ref_bands:
                        result[wb_idx] = _band_intensity(ref_bands[0])
                        continue
                # Fallback: highest intensity band
                result[wb_idx] = max(_band_intensity(b) for b in lane_bands)
            else:
                # total mode: sum all bands
                result[wb_idx] = sum(_band_intensity(b) for b in lane_bands)

        return result

    def get_summary_df(self, lane_types: dict[int, str] | None = None) -> pd.DataFrame:
        """Return a summary DataFrame for display in the UI.

        Columns: ponceau_lane, wb_lane, raw_intensity, loading_factor
        """
        ponceau_factors = self.get_loading_factors(lane_types=lane_types)
        lane_types = lane_types or {}

        records = []
        for pon_idx, factor in ponceau_factors.items():
            lane_bands = [
                b
                for b in self.state.bands
                if b.lane_index == pon_idx and getattr(b, "selected", True)
            ]
            if self.mode == "total":
                raw = sum(_band_intensity(b) for b in lane_bands)
            else:
                ref_idx = self.ref_band_indices.get(pon_idx, 0)
                ref_bands = [b for b in lane_bands if b.band_index == ref_idx]
                raw = _band_intensity(ref_bands[0]) if ref_bands else 0.0

            wb_idx = self.lane_mapping.get(pon_idx, pon_idx)
            records.append(
                {
                    "ponceau_lane": pon_idx,
                    "wb_lane": wb_idx,
                    "raw_intensity": round(raw, 4),
                    "loading_factor": round(factor, 4),
                }
            )

        return pd.DataFrame(records)
