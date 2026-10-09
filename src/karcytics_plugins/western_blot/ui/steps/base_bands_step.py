"""Base class for band detection steps to unify canvas, profiles, and UI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from karcytics_sdk.plugin import WizardPanel, WizardStep
from karcytics_sdk.plugin.theme_fallback import theme_manager
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .base_step import BaseStepWidget

if TYPE_CHECKING:
    from ..lane_profile_dialog import LaneProfileDialog

logger = logging.getLogger(__name__)


class BaseBandsStep(WizardStep):
    """Abstract base step for band detection (handles WB and Ponceau)."""

    _step_title = "Step 3: Detect Bands"
    _step_subtitle = "Configure parameters to detect bands in each lane."
    _detect_btn_text = "🔬  Detect Bands"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scientific_mode_enabled = True  # Default state

    def get_analyzer(self):
        raise NotImplementedError

    def _get_lane_types(self) -> dict[int, str]:
        return {}

    def _update_subclass_ui(self):
        pass

    # ── UI Construction ──────────────────────────────────────────────────

    def build_page(self, panel: WizardPanel) -> QWidget:
        self._panel = panel
        self._active_dialog: LaneProfileDialog | None = None

        page = BaseStepWidget(title=self._step_title, subtitle=self._step_subtitle)
        layout = page.content_layout
        layout.setSpacing(10)

        # Allow subclass to add specific settings (like SNR, Baseline, or Chart)
        self._build_extra_ui(layout, panel)

        # ── ADD THIS NEW TOGGLE HERE ──
        self.chk_auto_snap = QCheckBox("Auto-snap manual bands to nearest peak")
        self.chk_auto_snap.setChecked(True)
        self.chk_auto_snap.setToolTip(
            "Checked: Clicks will snap to the highest local intensity.\n"
            "Unchecked: Clicks and drags will place bands exactly where you draw them."
        )
        layout.addWidget(self.chk_auto_snap)
        # ──────────────────────────────

        # Common Actions
        self.btn_detect = QPushButton(self._detect_btn_text)
        theme_manager.apply_style(
            self.btn_detect,
            "QPushButton { background-color: {ACCENT_PRIMARY}; color: {BG_DARKEST};"
            " border: none; border-radius: 6px; padding: 8px 16px; font-weight: 600; }"
            "QPushButton:hover { background-color: {ACCENT_PRIMARY_HOVER}; }"
            "QPushButton:pressed { background-color: {ACCENT_PRIMARY_HOVER}; }"
            "QPushButton:disabled { background-color: {BG_MEDIUM}; color: {FG_DISABLED}; }",
        )
        self.btn_detect.setMinimumHeight(36)
        self.btn_detect.clicked.connect(self._detect_bands)
        layout.addWidget(self.btn_detect)

        self.btn_profiles = QPushButton("📈  View Lane Profiles")
        self.btn_profiles.setMinimumHeight(34)
        self.btn_profiles.setToolTip(
            "Opens a plot showing the lane density, baseline, and detected peaks."
        )
        self.btn_profiles.clicked.connect(self._show_profiles)
        layout.addWidget(self.btn_profiles)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("subtitle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setMinimumHeight(18)
        layout.addWidget(self.lbl_status)

        # Allow subclass to add settings after actions (like the Ponceau Chart)
        self._build_post_actions_ui(layout, panel)

        layout.addStretch()
        return self._scroll(page)

    def _build_extra_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        pass

    def _build_post_actions_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        pass

    def _detect_bands(self) -> None:
        raise NotImplementedError

    def _on_profiles_closed(self) -> None:
        pass

    # ── Profile Dialog ────────────────────────────────────────────────

    def _show_profiles(self) -> None:
        from ..lane_profile_dialog import LaneProfileDialog

        analyzer = self.get_analyzer()
        if not analyzer.state.profiles:
            try:
                self._detect_bands()
            except Exception as e:
                self._panel.status_message.emit(f"Could not compute profiles: {e}")
                return

        dialog = LaneProfileDialog(analyzer.state, None)
        self._active_dialog = dialog

        dialog.profile_hovered.connect(lambda li, y: self._panel.profile_hovered.emit(li, y))
        dialog.profile_clicked.connect(self._on_profile_clicked)
        dialog.profile_range_selected.connect(self._on_profile_range_selected)
        dialog.profile_band_removed.connect(self._on_profile_band_removed)

        dialog.exec()
        self._active_dialog = None
        self._on_profiles_closed()

    # ── Sync & History ─────────────────────────────────────────────────

    def _sync_canvas_and_history(self):
        analyzer = self.get_analyzer()

        # 1. Invalidate old results! This forces the UI to hide the results widget.
        analyzer.state.results_df = None

        # ─── INJECT SCIENTIFIC ALIGNMENT ─────────────────────────────────
        if self.scientific_mode_enabled and analyzer.state.bands:
            from ...analysis.actions.apply_scientific_bands import ApplyScientificBandsAction

            action = ApplyScientificBandsAction(
                state=analyzer.state,
                lane_types=self._get_lane_types(),
                tolerance_px=15.0,  # Tweak this tolerance as needed
            )
            action.execute()

            # --- THE FIX: Calculate math for auto-aligned "ghost" bands ---
            import numpy as np

            for b in analyzer.state.bands:
                # If intensity is suspiciously low (default 0.0 or 0.1), it's a ghost band missing its calculus
                if getattr(b, "integrated_intensity", 0) <= 0.1 and analyzer.state.profiles:
                    if b.lane_index < len(analyzer.state.profiles):
                        prof = analyzer.state.profiles[b.lane_index]
                        base = analyzer.state.baselines[b.lane_index]
                        corr = np.maximum(prof - base, 0)

                        # Find the peak height and exact bounds
                        y = max(0, min(int(b.position), len(corr) - 1))
                        b.peak_height = float(corr[y])
                        b.raw_height = float(prof[y])

                        hw = int(max(1, b.width / 2))
                        y0 = max(0, y - hw)
                        y1 = min(len(corr) - 1, y + hw)

                        # Integrate the actual area under the curve for this specific lane
                        b.integrated_intensity = max(0.1, float(np.sum(corr[y0 : y1 + 1])))

                        # Estimate SNR
                        noise = np.std(base) if np.std(base) > 0 else 1.0
                        b.snr = max(0.0, float(b.peak_height / noise))
        # ─────────────────────────────────────────────────────────────────

        self._panel.bands_detected.emit(analyzer.state.bands, analyzer.state.lanes)

        lane_types = self._get_lane_types()
        sample_bands = [
            b
            for b in analyzer.state.bands
            if lane_types.get(b.lane_index, "Sample") == "Sample" and getattr(b, "selected", True)
        ]
        self._panel.selected_bands_changed.emit(sample_bands)
        self._update_subclass_ui()

        if getattr(self, "_active_dialog", None) is not None:
            self._active_dialog._update_plot()

        self._panel.state_changed.emit()

    # ── Manual Interactions ──────────────────────────────────────────

    def _on_profile_band_removed(self, lane_idx: int, y_pos: float) -> None:
        if self._remove_linked_band_logic(lane_idx, y_pos):
            self._panel.status_message.emit("Linked band cluster removed.")
            self._sync_canvas_and_history()

    def on_band_right_clicked(self, lane_idx: int, y_pos: float, panel: WizardPanel) -> None:
        if self._remove_linked_band_logic(lane_idx, y_pos):
            panel.status_message.emit("Linked band cluster removed.")
            self._sync_canvas_and_history()

    # ── Manual Interactions ──────────────────────────────────────────

    def _on_profile_clicked(self, lane_idx: int, y_pos: float, auto_snap: bool = True) -> None:
        analyzer = self.get_analyzer()
        if lane_idx < 0 or lane_idx >= len(analyzer.state.lanes) or y_pos < 0:
            return
        try:
            # FORCE IT TO USE THE CHECKBOX
            actual_snap = (
                self.chk_auto_snap.isChecked() if hasattr(self, "chk_auto_snap") else False
            )
            band = analyzer.add_manual_band(lane_idx, int(round(y_pos)), auto_snap=actual_snap)
            if band:
                self._panel.status_message.emit(f"Added manual band in lane {lane_idx + 1}")
                self._sync_canvas_and_history()
            else:
                self._panel.status_message.emit("No clear peak near click.")
        except Exception as e:
            self._panel.status_message.emit(f"Error adding band: {e}")

    def _on_profile_range_selected(
        self, lane_idx: int, y_start: float, y_end: float, auto_snap: bool = True
    ) -> None:
        try:
            # FORCE IT TO USE THE CHECKBOX
            actual_snap = self.chk_auto_snap.isChecked()
            band = self.get_analyzer().add_manual_band_range(
                lane_idx, y_start, y_end, auto_snap=actual_snap
            )
            if band:
                self._panel.status_message.emit(f"Added band from range in lane {lane_idx + 1}")
                self._sync_canvas_and_history()
        except Exception as e:
            self._panel.status_message.emit(f"Error adding band range: {e}")

    def on_peak_pick_requested(
        self, x: float, y: float, panel: WizardPanel, auto_snap: bool = True
    ) -> None:
        analyzer = self.get_analyzer()
        lane = next(
            (
                ln
                for ln in analyzer.state.lanes
                if ln.x_start <= x <= ln.x_end and ln.y_start <= y <= ln.y_end
            ),
            None,
        )
        if not lane:
            return
        try:
            # FORCE IT TO USE THE CHECKBOX
            actual_snap = self.chk_auto_snap.isChecked()
            band = analyzer.add_manual_band(
                lane.index, int(round(y - lane.y_start)), auto_snap=actual_snap
            )
            if band:
                self._sync_canvas_and_history()
            else:
                panel.status_message.emit("No clear peak near click.")
        except Exception as e:
            panel.status_message.emit(f"Error adding manual band: {e}")

    def on_canvas_range_selected(
        self,
        lane_idx: int,
        y_start: float,
        y_end: float,
        panel: WizardPanel,
        auto_snap: bool = True,
    ) -> None:
        try:
            # FORCE IT TO USE THE CHECKBOX
            actual_snap = self.chk_auto_snap.isChecked()
            band = self.get_analyzer().add_manual_band_range(
                lane_idx, y_start, y_end, auto_snap=actual_snap
            )
            if band:
                self._sync_canvas_and_history()
        except Exception as e:
            panel.status_message.emit(f"Error dragging band range: {e}")

    def _remove_linked_band_logic(self, lane_idx: int, y_pos: float) -> bool:
        """Handles removing a band and its linked counterparts if scientific mode is on."""
        analyzer = self.get_analyzer()

        if self.scientific_mode_enabled:
            # 1. Find the exact band the user clicked on (with a 10px tolerance)
            target_band = None
            min_dist = float("inf")
            for b in analyzer.state.bands:
                if b.lane_index == lane_idx:
                    dist = abs(b.position - y_pos)
                    if dist <= 10.0 and dist < min_dist:
                        target_band = b
                        min_dist = dist

            # 2. If it is part of a scientific cluster, delete the whole cluster across all lanes
            if target_band and getattr(target_band, "matched_band", None) is not None:
                match_id = target_band.matched_band
                # Overwrite state with all bands EXCEPT the ones matching this cluster ID
                analyzer.state.bands = [
                    b for b in analyzer.state.bands if getattr(b, "matched_band", None) != match_id
                ]
                return True

        # 3. Fallback for standard individual deletion
        return analyzer.remove_band_at(lane_idx, y_pos)

    def on_enter(self) -> None:
        """Architectural Pillar 1 & 2:
        Strict Unidirectional State Hydration for Bands.
        """
        if not hasattr(self, "_panel") or not self._panel:
            return

        analyzer = self.get_analyzer()
        if not analyzer or not analyzer.state:
            return

        # 1. Explicit Canvas Context Switching (Pillar 2)
        if analyzer.state.processed_image is not None:
            self._panel.image_changed.emit(analyzer.state.processed_image)
        elif analyzer.state.original_image is not None:
            self._panel.image_changed.emit(analyzer.state.original_image)

        # 2. Broadcast Lanes (so the lane boundaries remain visible)
        if analyzer.state.lanes:
            self._panel.lanes_detected.emit(analyzer.state.lanes)
        else:
            self._panel.lanes_detected.emit([])

        # 3. Broadcast Bands (so the detected bands appear)
        if analyzer.state.bands:
            self._panel.bands_detected.emit(analyzer.state.bands, analyzer.state.lanes)
        else:
            self._panel.bands_detected.emit([], [])

        # 4. Trigger the subclass UI refresh (e.g., Ponceau's reference band status)
        self._update_subclass_ui()
