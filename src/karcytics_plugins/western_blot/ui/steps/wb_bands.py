"""Western Blot — Step 3: Band Detection."""

from __future__ import annotations

import logging

from karcytics_sdk.plugin import WizardPanel
from karcytics_sdk.plugin.theme_fallback import Colors
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from .base_bands_step import BaseBandsStep

logger = logging.getLogger(__name__)


class WBBandsStep(BaseBandsStep):
    """Detect bands in each lane and allow manual correction."""

    label = "Bands"
    _step_title = "Step 3: Detect Bands"
    _step_subtitle = "Configure parameters to detect bands in each lane."
    _detect_btn_text = "🔬  Detect Bands"

    def get_analyzer(self):
        return self._panel.analyzer

    def _get_lane_types(self) -> dict[int, str]:
        from .wb_lanes import WBLanesStep

        for step in self._panel._steps:
            if isinstance(step, WBLanesStep):
                return step.get_lane_types()
        return {}

    def _build_extra_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        # ── Detection parameters ──────────────────────────────────────
        band_group = QGroupBox("Detection Parameters")
        band_layout = QVBoxLayout(band_group)
        band_layout.setSpacing(8)

        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(1.0, 20.0)
        self.spin_snr.setValue(3.0)
        self.spin_snr.setSingleStep(0.5)
        self.spin_snr.setToolTip(
            "Signal-to-noise ratio threshold.\n• 2.0 = lenient  • 3.0 = default  • 5.0 = strict"
        )
        band_layout.addLayout(self._row("Min SNR:", self.spin_snr))

        self.spin_peak_distance = QSpinBox()
        self.spin_peak_distance.setRange(3, 100)
        self.spin_peak_distance.setValue(10)
        self.spin_peak_distance.setSuffix(" px")
        self.spin_peak_distance.setToolTip("Minimum distance between adjacent bands in pixels.")
        band_layout.addLayout(self._row("Min band spacing:", self.spin_peak_distance))

        self.spin_max_width = QSpinBox()
        self.spin_max_width.setRange(5, 500)
        self.spin_max_width.setValue(80)
        self.spin_max_width.setSuffix(" px")
        self.spin_max_width.setToolTip(
            "Maximum allowed band width — wider peaks are likely background artifacts."
        )
        band_layout.addLayout(self._row("Max band width:", self.spin_max_width))

        self.spin_min_width = QSpinBox()
        self.spin_min_width.setRange(1, 50)
        self.spin_min_width.setValue(3)
        self.spin_min_width.setSuffix(" px")
        self.spin_min_width.setToolTip(
            "Minimum band width — narrower peaks are likely noise spikes."
        )
        band_layout.addLayout(self._row("Min band width:", self.spin_min_width))

        self.spin_edge_margin = QDoubleSpinBox()
        self.spin_edge_margin.setRange(0.0, 25.0)
        self.spin_edge_margin.setValue(5.0)
        self.spin_edge_margin.setSuffix(" %")
        self.spin_edge_margin.setToolTip(
            "% of lane height at top/bottom to ignore (rotation/crop edge artifacts)."
        )
        band_layout.addLayout(self._row("Edge margin:", self.spin_edge_margin))
        layout.addWidget(band_group)

        # ── Baseline ──────────────────────────────────────────────────
        baseline_group = QGroupBox("Baseline Estimation")
        baseline_layout = QVBoxLayout(baseline_group)
        baseline_layout.setSpacing(8)

        self.combo_baseline = QComboBox()
        self.combo_baseline.addItems(["Rolling Ball", "Linear"])
        self.combo_baseline.setToolTip(
            "Rolling Ball: smooth background subtraction (recommended).\nLinear: straight-line baseline between peak valleys."
        )
        baseline_layout.addLayout(self._row("Method:", self.combo_baseline))

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(0, 200)
        self.spin_radius.setValue(0)
        self.spin_radius.setSpecialValueText("Auto")
        self.spin_radius.setSuffix(" px")
        self.spin_radius.setToolTip(
            "Rolling ball radius.\n'Auto' (0) = 40% of lane height per-lane."
        )
        baseline_layout.addLayout(self._row("Radius:", self.spin_radius))
        layout.addWidget(baseline_group)

        # ── Scientific Alignment ──────────────────────────────────────
        sci_group = QGroupBox("Scientific Alignment")
        sci_layout = QVBoxLayout(sci_group)

        self.chk_scientific = QCheckBox("Enforce unified band boundaries across sample lanes")
        self.chk_scientific.setChecked(self.scientific_mode_enabled)
        self.chk_scientific.setToolTip(
            "Aligns bands across lanes and backfills missing regions to ensure\n"
            "densitometry comparisons use identical ROI boundaries."
        )
        self.chk_scientific.toggled.connect(self._on_scientific_toggled)
        sci_layout.addWidget(self.chk_scientific)
        layout.addWidget(sci_group)

        # ── Manual peak picking ───────────────────────────────────────
        manual_group = QGroupBox("ImageJ-style Peak Picking")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setSpacing(6)

        self.chk_manual_pick = QCheckBox(
            "Manual peak picking (for messy blots with bad auto-detection)"
        )
        self.chk_manual_pick.setChecked(False)
        self.chk_manual_pick.setToolTip(
            "Compute profiles/baselines only, then click on bands to quantify.\nMirrors ImageJ's gel workflow."
        )
        self.chk_manual_pick.toggled.connect(self._on_manual_pick_toggled)
        manual_layout.addWidget(self.chk_manual_pick)

        hint = QLabel(
            "Workflow: enable → click 'Detect Bands' (profiles only) → click bands in the image."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        hint.setMinimumHeight(32)
        manual_layout.addWidget(hint)
        layout.addWidget(manual_group)

    def on_enter(self) -> None:
        pass

    def on_next(self, panel: WizardPanel) -> bool:
        if not panel.analyzer.state.bands:
            self._detect_bands()
        return bool(panel.analyzer.state.bands)

    # ── Detection ─────────────────────────────────────────────────────

    def _detection_params(self) -> dict:
        baseline_text = self.combo_baseline.currentText()
        return dict(
            min_peak_height=0.02,
            min_peak_distance=self.spin_peak_distance.value(),
            min_snr=self.spin_snr.value(),
            max_band_width=self.spin_max_width.value(),
            min_band_width=self.spin_min_width.value(),
            edge_margin_percent=self.spin_edge_margin.value(),
            baseline_method="rolling_ball" if "Rolling" in baseline_text else "linear",
            baseline_radius=self.spin_radius.value(),
        )

    def _detect_bands(self) -> None:
        from karcytics_sdk.plugin import task_scheduler

        try:
            analyzer = self._panel.analyzer
            params = self._detection_params()
            manual_pick = self.chk_manual_pick.isChecked()

            # Setup background task
            analyzer.current_task_type = "detect_bands"
            analyzer.current_task_params = {
                **params,
                "manual_pick": manual_pick,
                "force_valleys_as_bands": None,
            }

            self.btn_detect.setEnabled(False)
            self.lbl_status.setText("⌛  Detecting bands...")
            self.lbl_status.setStyleSheet(f"color: {Colors.FG_PRIMARY};")

            worker = task_scheduler.submit(analyzer, analyzer.state)
            task_id = getattr(worker, "task_id", "")

            def _on_finished(tid, results):
                if tid != task_id:
                    return
                task_scheduler.task_finished.disconnect(_on_finished)
                task_scheduler.task_error.disconnect(_on_error)

                # Unpack results back into analyzer state
                analyzer.state.bands = results.get("bands", [])
                analyzer.state.profiles = results.get("profiles", [])
                analyzer.state.baselines = results.get("baselines", [])
                analyzer.state.lane_orientations = results.get("lane_orientations", [])
                analyzer.state.detection_image = results.get("detection_image")

                self.btn_detect.setEnabled(True)
                bands = analyzer.state.bands
                lane_types = self._get_lane_types()
                sample_bands = [
                    b for b in bands if lane_types.get(b.lane_index, "Sample") == "Sample"
                ]

                # Per-lane summary
                counts: dict[int, int] = {}
                for b in bands:
                    counts.setdefault(b.lane_index, 0)
                    counts[b.lane_index] += 1
                summary = " | ".join(
                    f"L{i + 1}: {n}{' [' + lane_types.get(i, 'S')[0] + ']' if lane_types.get(i, 'Sample') != 'Sample' else ''}"
                    for i, n in sorted(counts.items())
                )

                if manual_pick:
                    self.lbl_status.setText(
                        f"✅  Profiles computed. Click bands in the image.\n{summary}"
                    )
                else:
                    self.lbl_status.setText(
                        f"✅  {len(bands)} bands ({len(sample_bands)} sample)\n{summary}"
                    )
                self.lbl_status.setStyleSheet(f"color: {Colors.SUCCESS};")
                self._panel.status_message.emit(
                    f"Detected {len(bands)} bands ({len(sample_bands)} sample)"
                )
                self._sync_canvas_and_history()

            def _on_error(tid, error_msg):
                if tid != task_id:
                    return
                task_scheduler.task_finished.disconnect(_on_finished)
                task_scheduler.task_error.disconnect(_on_error)

                self.btn_detect.setEnabled(True)
                self.lbl_status.setText(f"❌  {error_msg}")
                self.lbl_status.setStyleSheet(f"color: {Colors.ACCENT_DANGER};")
                logger.error(f"Band detection task error: {error_msg}")

            task_scheduler.task_finished.connect(_on_finished)
            task_scheduler.task_error.connect(_on_error)

        except Exception as e:
            self.btn_detect.setEnabled(True)
            self.lbl_status.setText(f"❌  {e}")
            self.lbl_status.setStyleSheet(f"color: {Colors.ACCENT_DANGER};")
            logger.exception("Band detection error during submission")

    def _on_manual_pick_toggled(self, enabled: bool) -> None:
        self._panel.peak_picking_enabled.emit(enabled)
        msg = (
            "Manual picking on — click 'Detect Bands' then click bands in the image."
            if enabled
            else "Manual picking disabled."
        )
        self._panel.status_message.emit(msg)

    def on_band_clicked(self, band, panel: WizardPanel) -> None:
        band.selected = True

        rw = getattr(panel, "_results_widget_ref", None)
        if rw is not None and hasattr(rw, "highlight_band_for_comparison"):
            rw.highlight_band_for_comparison(band)

        panel.status_message.emit(
            f"Band selected: Lane {band.lane_index + 1}, pos {band.position}, "
            f"intensity {band.integrated_intensity:.2f}, SNR {band.snr:.1f}  —  "
            f"click another band to compare"
        )

    def _on_scientific_toggled(self, checked: bool) -> None:
        self.scientific_mode_enabled = checked
        # If toggled, re-run detection so we get clean, non-mutated bands if turned off
        if self._panel.analyzer.state.profiles:
            self._detect_bands()
