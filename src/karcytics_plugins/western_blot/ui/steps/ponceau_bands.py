"""Ponceau Stain — Step 3: Band Detection & Loading Factors."""

from __future__ import annotations

import logging

from karcytics_sdk.plugin import WizardPanel
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .base_bands_step import BaseBandsStep

logger = logging.getLogger(__name__)


class _FactorChart(QWidget):
    """Mini bar chart showing per-lane Ponceau loading factors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.fig = None
        self.canvas = None
        self.ax = None

    def _ensure_canvas(self):
        if self.canvas is not None:
            return

        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(5, 2.2), dpi=90)
        self.fig.patch.set_facecolor(Colors.BG_DARK)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout().addWidget(self.canvas)
        self.canvas.setStyleSheet(f"background-color: {Colors.BG_DARK};")

    def _draw_empty(self) -> None:
        self._ensure_canvas()
        self.ax.clear()
        self.ax.set_facecolor(Colors.BG_DARK)
        self.ax.text(
            0.5,
            0.5,
            "Run band detection to see loading factors",
            ha="center",
            va="center",
            color=Colors.FG_SECONDARY,
            fontsize=9,
            transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_factors(
        self,
        factors: dict[int, float],
        num_lanes: int,
        label_prefix: str = "WB",
    ) -> None:
        self._ensure_canvas()
        self.ax.clear()
        self.ax.set_facecolor(Colors.BG_DARK)

        lanes = list(range(num_lanes))
        values = [factors.get(i, 1.0) for i in lanes]
        labels = [f"{label_prefix} {i + 1}" for i in lanes]

        colors = [
            Colors.ACCENT_PRIMARY
            if abs(v - 1.0) < 0.15
            else Colors.ACCENT_WARNING
            if abs(v - 1.0) < 0.35
            else Colors.ACCENT_DANGER
            for v in values
        ]

        bars = self.ax.bar(
            range(num_lanes), values, color=colors, edgecolor="none", width=0.6, alpha=0.9
        )

        self.ax.axhline(1.0, color=Colors.FG_SECONDARY, linewidth=0.8, linestyle="--", alpha=0.6)

        for bar, val in zip(bars, values, strict=False):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=Colors.FG_SECONDARY,
            )

        self.ax.set_xticks(range(num_lanes))
        self.ax.set_xticklabels(labels, fontsize=8, color=Colors.FG_SECONDARY)
        self.ax.set_ylabel("Loading factor", fontsize=8, color=Colors.FG_SECONDARY)
        title = (
            "Ponceau loading factors → WB lanes"
            if label_prefix == "WB"
            else "Ponceau loading factors (WB lanes not yet detected)"
        )
        self.ax.set_title(title, fontsize=9, color=Colors.FG_PRIMARY)
        self.ax.tick_params(colors=Colors.FG_SECONDARY)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            self.ax.spines[spine].set_color(Colors.BORDER)

        self.fig.tight_layout()
        self.canvas.draw()


class PonceauBandsStep(BaseBandsStep):
    """Detect Ponceau bands and preview loading factors."""

    label = "Pon. Bands"
    _step_title = "Step 3: Ponceau Bands"
    _step_subtitle = "Detect Ponceau bands and compute per-lane loading factors."
    _detect_btn_text = "🔬  Detect Ponceau Bands"

    def get_analyzer(self):
        return self._panel.ponceau_analyzer

    def _build_extra_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        # Quantification mode
        mode_group = QGroupBox("Quantification Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(6)

        mode_hint = QLabel(
            "Reference band: pick one prominent band per lane (matches ImageJ protocol).\n"
            "Total lane: sum all detected bands — more statistically robust."
        )
        mode_hint.setWordWrap(True)
        mode_hint.setObjectName("subtitle")
        mode_layout.addWidget(mode_hint)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Reference band (per-lane)", "Total lane intensity"])
        self.combo_mode.setCurrentIndex(0)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addLayout(self._row("Mode:", self.combo_mode))
        layout.addWidget(mode_group)

        # Reference band selection status
        self._ref_group = QGroupBox("Reference Band Selection")
        ref_layout = QVBoxLayout(self._ref_group)
        ref_layout.setSpacing(6)

        ref_instr = QLabel(
            "After detecting bands, click a band on the image canvas to set it as the reference for that lane."
        )
        ref_instr.setWordWrap(True)
        ref_instr.setObjectName("subtitle")
        ref_layout.addWidget(ref_instr)

        self.lbl_ref_status = QLabel("(detect bands first)")
        self.lbl_ref_status.setWordWrap(True)
        self.lbl_ref_status.setObjectName("subtitle")
        self.lbl_ref_status.setMinimumHeight(18)
        ref_layout.addWidget(self.lbl_ref_status)

        layout.addWidget(self._ref_group)

        # Detection settings
        det_group = QGroupBox("Detection Settings")
        det_layout = QVBoxLayout(det_group)
        det_layout.setSpacing(8)

        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(1.0, 10.0)
        self.spin_snr.setValue(2.0)
        self.spin_snr.setSingleStep(0.5)
        det_layout.addLayout(self._row("Min SNR:", self.spin_snr))

        self.spin_min_distance = QSpinBox()
        self.spin_min_distance.setRange(3, 100)
        self.spin_min_distance.setValue(8)
        self.spin_min_distance.setSuffix(" px")
        det_layout.addLayout(self._row("Min spacing:", self.spin_min_distance))

        layout.addWidget(det_group)

        # ── Scientific Alignment ──────────────────────────────────────
        sci_group = QGroupBox("Scientific Alignment")
        sci_layout = QVBoxLayout(sci_group)

        self.chk_scientific = QCheckBox("Enforce unified band boundaries across lanes")
        # Default to False for Ponceau since total lane intensity doesn't need strict alignment
        self.chk_scientific.setChecked(self.scientific_mode_enabled)
        self.chk_scientific.toggled.connect(self._on_scientific_toggled)
        sci_layout.addWidget(self.chk_scientific)
        layout.addWidget(sci_group)

    def _build_post_actions_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        chart_group = QGroupBox("Loading Factors Preview")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _FactorChart()
        chart_layout.addWidget(self._chart)

        self.lbl_chart_hint = QLabel(
            "Green bars ≈ 1.0 (well-loaded).  "
            "Amber/red bars indicate unequal loading — "
            "Ponceau normalization will correct for this."
        )
        self.lbl_chart_hint.setWordWrap(True)
        self.lbl_chart_hint.setObjectName("subtitle")
        chart_layout.addWidget(self.lbl_chart_hint)
        layout.addWidget(chart_group)

    def on_enter(self) -> None:
        analyzer = self._panel.ponceau_analyzer
        if analyzer is None:
            return

        if analyzer.state.processed_image is not None:
            self._panel.image_changed.emit(analyzer.state.processed_image)

        if analyzer.state.lanes:
            self._panel.lanes_detected.emit(analyzer.state.lanes)

        if analyzer.state.bands:
            self._panel.bands_detected.emit(analyzer.state.bands, analyzer.state.lanes)
            self._update_chart()
            self._refresh_ref_band_status()

    def on_band_clicked(self, band, panel: WizardPanel) -> None:
        if panel.ponceau_analyzer is None:
            return

        mode = panel.ponceau_analyzer.mode
        lane_idx = band.lane_index

        if mode == "reference_band":
            panel.ponceau_analyzer.ref_band_indices[lane_idx] = band.band_index
            panel.status_message.emit(
                f"Ponceau reference band set: Lane {lane_idx + 1}, "
                f"band {band.band_index + 1}, "
                f"intensity {band.integrated_intensity:.3f}"
            )
            self._update_chart()
            self._refresh_ref_band_status()
            self._panel.state_changed.emit()
        else:
            panel.status_message.emit(
                f"Ponceau band: Lane {lane_idx + 1}, "
                f"pos {band.position}, "
                f"intensity {band.integrated_intensity:.3f}"
            )

    def _refresh_ref_band_status(self) -> None:
        if not hasattr(self, "lbl_ref_status"):
            return
        analyzer = self._panel.ponceau_analyzer
        if analyzer is None or analyzer.mode != "reference_band":
            self.lbl_ref_status.setText("")
            return

        lanes = analyzer.state.lanes
        if not lanes:
            return

        lines = []
        all_set = True
        for lane in lanes:
            idx = lane.index
            lane_bands = [b for b in analyzer.state.bands if b.lane_index == idx]
            if not lane_bands:
                lines.append(f"Lane {idx + 1}: no bands (will use total lane fallback)")
                all_set = False
                continue
            ref_idx = analyzer.ref_band_indices.get(idx)
            if ref_idx is None:
                lines.append(f"Lane {idx + 1}: ⚠️ not set — click a band on the image")
                all_set = False
            else:
                ref_bands = [b for b in lane_bands if b.band_index == ref_idx]
                if ref_bands:
                    b = ref_bands[0]
                    lines.append(
                        f"Lane {idx + 1}: ✅ band {ref_idx + 1} "
                        f"(pos {b.position}, int {b.integrated_intensity:.3f})"
                    )
                else:
                    lines.append(f"Lane {idx + 1}: ⚠️ band {ref_idx + 1} not found")
                    all_set = False

        self.lbl_ref_status.setText("\n".join(lines))
        if all_set:
            self.lbl_ref_status.setStyleSheet(f"color: {Colors.SUCCESS};")
        elif any("\u26a0" in line for line in lines):
            self.lbl_ref_status.setStyleSheet(f"color: {Colors.ACCENT_WARNING};")
        else:
            self.lbl_ref_status.setStyleSheet(f"color: {Colors.FG_SECONDARY};")

    def on_next(self, panel: WizardPanel) -> bool:
        if not panel.ponceau_analyzer.state.bands:
            self._detect_bands()
        if not panel.ponceau_analyzer.state.bands:
            panel.status_message.emit("No Ponceau bands detected — check contrast/SNR settings.")
            return False
        self._sync_mode(panel)
        return True

    def _detect_bands(self) -> None:
        from karcytics_sdk.plugin import task_scheduler

        try:
            analyzer = self._panel.ponceau_analyzer
            analyzer.ref_band_indices.clear()

            params = {
                "min_snr": self.spin_snr.value(),
                "min_peak_distance": self.spin_min_distance.value(),
                "force_valleys_as_bands": None,
            }

            # Setup background task
            analyzer.current_task_type = "detect_bands"
            analyzer.current_task_params = params

            self.btn_detect.setEnabled(False)
            self.lbl_status.setText("⌛  Detecting Ponceau bands...")
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
                n = len(analyzer.state.bands)
                self.lbl_status.setText(f"✅  {n} Ponceau bands detected")
                self.lbl_status.setStyleSheet(f"color: {Colors.SUCCESS};")
                self._panel.status_message.emit(f"Ponceau: {n} bands detected")

                self._sync_mode(self._panel)
                self._sync_canvas_and_history()
                self._update_chart()

            def _on_error(tid, error_msg):
                if tid != task_id:
                    return
                task_scheduler.task_finished.disconnect(_on_finished)
                task_scheduler.task_error.disconnect(_on_error)

                self.btn_detect.setEnabled(True)
                self.lbl_status.setText(f"❌  {error_msg}")
                self.lbl_status.setStyleSheet(f"color: {Colors.ACCENT_DANGER};")
                logger.error(f"Ponceau detection task error: {error_msg}")

            task_scheduler.task_finished.connect(_on_finished)
            task_scheduler.task_error.connect(_on_error)

        except Exception as e:
            self.btn_detect.setEnabled(True)
            self.lbl_status.setText(f"❌  {e}")
            self.lbl_status.setStyleSheet(f"color: {Colors.ACCENT_DANGER};")
            logger.exception("Ponceau band detection error during submission")

    def _on_mode_changed(self, _idx: int) -> None:
        self._sync_mode(self._panel)
        if self._panel.ponceau_analyzer.state.bands:
            self._update_chart()
        self._panel.state_changed.emit()

    def _sync_mode(self, panel: WizardPanel) -> None:
        mode = "reference_band" if self.combo_mode.currentIndex() == 0 else "total"
        panel.ponceau_analyzer.mode = mode
        self._ref_group.setVisible(mode == "reference_band")
        self._refresh_ref_band_status()

    def _update_chart(self) -> None:
        try:
            nb_wb = len(self._panel.analyzer.state.lanes)
            if nb_wb > 0:
                factors = self._panel.ponceau_analyzer.get_wb_loading_factors(nb_wb)
                self._chart.plot_factors(factors, nb_wb, label_prefix="WB")
            else:
                factors = self._panel.ponceau_analyzer.get_loading_factors()
                nb_pon = len(self._panel.ponceau_analyzer.state.lanes)
                if nb_pon > 0 and factors:
                    full = {i: factors.get(i, 1.0) for i in range(nb_pon)}
                    self._chart.plot_factors(full, nb_pon, label_prefix="Pon.")
        except Exception as e:
            logger.warning("Could not update Ponceau chart: %s", e)

    def _update_subclass_ui(self):
        """Called automatically by BaseBandsStep whenever the user manually edits the canvas."""
        analyzer = self.get_analyzer()

        if getattr(analyzer, "mode", "total") == "reference_band":
            # 1. Map out surviving bands
            existing_bands = {(b.lane_index, b.band_index) for b in analyzer.state.bands}
            refs_removed = False

            # 2. Purge ghosts
            for lane_idx, ref_idx in list(analyzer.ref_band_indices.items()):
                if (lane_idx, ref_idx) not in existing_bands:
                    del analyzer.ref_band_indices[lane_idx]
                    refs_removed = True

            if refs_removed:
                self._panel.status_message.emit(
                    "⚠️ Reference band was removed! Please click a new reference band for the affected lane."
                )

        # 3. Force the UI visuals to refresh immediately
        self._update_chart()
        self._refresh_ref_band_status()

        # 4. ── THE CRUCIAL MISSING LINK ──
        # Tell the Results Step that the biological math has been permanently altered!
        # (We trigger this on ANY edit, because even in 'Total Lane' mode,
        # adding/deleting a band changes the total sum!)
        if self._panel:
            self._panel.state_changed.emit()

    def _on_scientific_toggled(self, checked: bool) -> None:
        self.scientific_mode_enabled = checked
        if self._panel.analyzer.state.profiles:
            # Re-run detection to cleanly reset bands when toggled off
            self._detect_bands()

    def _get_lane_types(self) -> dict[int, str]:
        """Translates the Ponceau mapping combos into lane types for the Action."""
        from .ponceau_lanes import PonceauLanesStep

        types = {}
        for step in self._panel._steps:
            if isinstance(step, PonceauLanesStep):
                for i, combo in enumerate(step._mapping_combos):
                    text = combo.currentText()
                    if text in ["Ladder", "Exclude"]:
                        types[i] = text
                    elif "Skip" in text:
                        types[i] = "Unmapped"
                    else:
                        types[i] = "Sample"  # It's a WB mapped lane
                break
        return types
