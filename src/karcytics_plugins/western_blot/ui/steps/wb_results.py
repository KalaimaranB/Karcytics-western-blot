"""Western Blot — Step 4: Results & Normalization."""

from __future__ import annotations

import logging

import pandas as pd
from karcytics_sdk.plugin import WizardPanel, WizardStep
from karcytics_sdk.plugin.dialogs import SaveWorkflowDialog
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class WBResultsStep(WizardStep):
    """Compute densitometry, apply normalization, emit results."""

    label = "Results"
    is_terminal = True

    def build_page(self, panel: WizardPanel) -> QWidget:
        self._panel = panel

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        # ── WB internal normalization ─────────────────────────────────
        norm_group = QGroupBox("WB Normalization")
        norm_layout = QVBoxLayout(norm_group)
        norm_layout.setSpacing(8)

        self.combo_ref_lane = QComboBox()
        self.combo_ref_lane.addItem("None (% of total)")
        self.combo_ref_lane.setToolTip(
            "Optionally pick a WB reference lane for band-based normalisation.\n"
            "Usually left as 'None' when Ponceau normalisation is used."
        )
        self.combo_ref_lane.currentIndexChanged.connect(lambda _: self._compute_results())
        self.combo_ref_lane.currentIndexChanged.connect(
            lambda _: self._panel.state_changed.emit()
        )  # <-- NEW
        norm_layout.addLayout(self._row("Reference lane:", self.combo_ref_lane))

        self.chk_normalize_one = QCheckBox("Set control lane to 1.0")
        self.chk_normalize_one.setChecked(False)
        self.chk_normalize_one.toggled.connect(lambda _: self._compute_results())
        self.chk_normalize_one.toggled.connect(
            lambda _: self._panel.state_changed.emit()
        )  # <-- NEW
        norm_layout.addWidget(self.chk_normalize_one)

        # ── THE NEW CHECKBOX ──
        self.chk_sum_lanes = QCheckBox("Sum multiple bands in the same lane (e.g., Total ERK1/2)")
        self.chk_sum_lanes.setChecked(False)
        self.chk_sum_lanes.toggled.connect(lambda _: self._compute_results())
        self.chk_sum_lanes.toggled.connect(lambda _: self._panel.state_changed.emit())
        norm_layout.addWidget(self.chk_sum_lanes)

        layout.addWidget(norm_group)

        # ── Ponceau status ────────────────────────────────────────────
        self._ponceau_group = QGroupBox("Ponceau Loading Normalisation")
        pon_layout = QVBoxLayout(self._ponceau_group)
        pon_layout.setSpacing(6)

        self.lbl_ponceau_status = QLabel("No Ponceau data — results will not be loading-corrected.")
        self.lbl_ponceau_status.setWordWrap(True)
        self.lbl_ponceau_status.setObjectName("subtitle")
        pon_layout.addWidget(self.lbl_ponceau_status)

        self.chk_use_ponceau = QCheckBox("Apply Ponceau loading correction")
        self.chk_use_ponceau.setChecked(True)
        self.chk_use_ponceau.setVisible(False)
        self.chk_use_ponceau.toggled.connect(lambda _: self._compute_results())
        self.chk_use_ponceau.toggled.connect(lambda _: self._panel.state_changed.emit())  # <-- NEW
        pon_layout.addWidget(self.chk_use_ponceau)

        layout.addWidget(self._ponceau_group)

        info = QLabel("Results update automatically in the right panel.")
        info.setObjectName("subtitle")
        info.setWordWrap(True)
        info.setMinimumHeight(32)
        layout.addWidget(info)

        # ── Workflow Management ───────────────────────────────────────
        workflow_group = QGroupBox("Session Management")
        workflow_layout = QVBoxLayout(workflow_group)

        self.btn_save_workflow = QPushButton("💾 Save Workflow")
        self.btn_save_workflow.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_save_workflow.setToolTip(
            "Save the final results and all step parameters as a reusable workflow."
        )
        self.btn_save_workflow.clicked.connect(lambda: self._on_save_workflow(panel))

        workflow_layout.addWidget(self.btn_save_workflow)
        layout.addWidget(workflow_group)

        layout.addStretch()
        return self._scroll(page)

    def on_enter(self) -> None:
        """Refresh Ponceau status and auto-compute when entering."""
        self._refresh_ponceau_status()

        # ── THE FIX: Dynamically populate the Reference Lane dropdown ──
        analyzer = self._panel.analyzer
        if analyzer and analyzer.state.lanes:
            current_idx = self.combo_ref_lane.currentIndex()

            self.combo_ref_lane.blockSignals(True)
            self.combo_ref_lane.clear()
            self.combo_ref_lane.addItem("None (% of total)")
            for i in range(len(analyzer.state.lanes)):
                self.combo_ref_lane.addItem(f"Lane {i + 1}")

            # Restore previous selection if valid so it doesn't reset on you
            if 0 <= current_idx < self.combo_ref_lane.count():
                self.combo_ref_lane.setCurrentIndex(current_idx)
            self.combo_ref_lane.blockSignals(False)
        # ───────────────────────────────────────────────────────────────

        self._compute_results()

        rw = getattr(self._panel, "_results_widget_ref", None)
        if rw is not None and hasattr(rw, "_spin_slots"):
            lane_types = self._get_lane_types()
            sample_lanes = [
                lane
                for lane in self._panel.analyzer.state.lanes
                if lane_types.get(lane.index, "Sample") == "Sample"
            ]
            num_samples = len(sample_lanes) if sample_lanes else 2

            # If the widget has fewer slots than we have lanes, automatically increase it!
            if rw._num_slots < num_samples:
                rw._spin_slots.setValue(num_samples)

        if analyzer and analyzer.state.processed_image is not None:
            self._panel.image_changed.emit(analyzer.state.processed_image)
            lanes = getattr(analyzer.state, "lanes", [])
            bands = getattr(analyzer.state, "bands", [])
            self._panel.lanes_detected.emit(lanes)
            self._panel.bands_detected.emit(bands, lanes)

    def on_next(self, panel: WizardPanel) -> bool:
        return False  # terminal

    def on_band_clicked(self, band, panel: WizardPanel) -> None:
        """Route canvas band clicks to toggle the band in the results widget."""
        rw = getattr(panel, "_results_widget_ref", None)
        if rw is not None and hasattr(rw, "toggle_band_selection"):
            added = rw.toggle_band_selection(band)
            if added:
                panel.status_message.emit(
                    f"Band added — Lane {band.lane_index + 1}, pos {band.position}px"
                )
            else:
                panel.status_message.emit(
                    f"Band removed — Lane {band.lane_index + 1}, pos {band.position}px"
                )

        # Recompute immediately when a band is toggled
        self._compute_results()
        self._panel.state_changed.emit()

    # ── Public ────────────────────────────────────────────────────────

    def update_ref_lane_combo(self, num_lanes: int) -> None:
        self.combo_ref_lane.blockSignals(True)
        self.combo_ref_lane.clear()
        self.combo_ref_lane.addItem("None (% of total)")
        for i in range(num_lanes):
            self.combo_ref_lane.addItem(f"Lane {i + 1}")
        self.combo_ref_lane.blockSignals(False)

    # ── Internal ──────────────────────────────────────────────────────

    def _refresh_ponceau_status(self) -> None:
        """Update the Ponceau status label based on available data."""
        ponceau = getattr(self._panel, "ponceau_analyzer", None)
        has_ponceau = ponceau is not None and ponceau.state.bands and ponceau.lane_mapping
        if has_ponceau:
            nb_mapped = len(ponceau.lane_mapping)
            mode = ponceau.mode.replace("_", " ")
            self.lbl_ponceau_status.setText(
                f"✅  Ponceau data available — {nb_mapped} lanes mapped "
                f"(mode: {mode}).\n"
                f"Loading factors will be applied to produce 'Ponceau Normalised' values."
            )
            self.lbl_ponceau_status.setStyleSheet(f"color: {Colors.SUCCESS};")
            self.chk_use_ponceau.setVisible(True)
        else:
            self.lbl_ponceau_status.setText(
                "No Ponceau data — results will show WB-only normalisation.\n"
                "Go back to complete the Ponceau stage to enable loading correction."
            )
            self.lbl_ponceau_status.setStyleSheet(f"color: {Colors.FG_SECONDARY};")
            self.chk_use_ponceau.setVisible(False)

    def _compute_results(self) -> None:
        try:
            lane_types = self._get_lane_types()
            nb_wb = len(self._panel.analyzer.state.lanes)

            # ── Step 1: Identify which bands to plot (Slots vs Defaults) ──
            rw = getattr(self._panel, "_results_widget_ref", None)
            bands_to_analyze = []
            has_user_slots = False

            if rw is not None and hasattr(rw, "_slots"):
                for i, band in enumerate(rw._slots):
                    if band is not None:
                        bands_to_analyze.append((i, band))
                        has_user_slots = True

            # Fallback: If no slots filled, pick brightest band per lane as virtual slots
            if not has_user_slots:
                for lane_idx in range(nb_wb):
                    lt = lane_types.get(lane_idx, "Sample")
                    if lt == "Exclude":
                        continue
                    lane_bands = [
                        b for b in self._panel.analyzer.state.bands if b.lane_index == lane_idx
                    ]
                    if lane_bands:
                        best = max(
                            lane_bands,
                            key=lambda x: (
                                x.integrated_intensity
                                if x.integrated_intensity > 1e-6
                                else x.peak_height
                            ),
                        )
                        # Use lane_idx as a pseudo slot_idx so colors map correctly on first load
                        bands_to_analyze.append((lane_idx, best))

            # ── Step 2: Get Ponceau raw intensity ──
            ponceau = getattr(self._panel, "ponceau_analyzer", None)
            use_ponceau = (
                self.chk_use_ponceau.isChecked()
                and self.chk_use_ponceau.isVisible()
                and ponceau is not None
                and ponceau.state.bands
                and ponceau.lane_mapping
            )
            ponceau_raw: dict[int, float] = {}
            if use_ponceau:
                ponceau_raw = ponceau.get_ponceau_raw_per_wb_lane(nb_wb)

            # ── Step 3: Compute math strictly per slot OR per lane sum ──
            records = []

            sum_lanes = getattr(self, "chk_sum_lanes", None) and self.chk_sum_lanes.isChecked()

            if sum_lanes and has_user_slots:
                # Group selected slots by lane and sum them!
                lane_sums: dict[int, float] = {}
                for _slot_idx, b in bands_to_analyze:
                    lane_idx = b.lane_index
                    wb_raw = (
                        float(b.integrated_intensity)
                        if float(b.integrated_intensity) > 1e-6
                        else float(b.peak_height)
                    )
                    lane_sums[lane_idx] = lane_sums.get(lane_idx, 0.0) + wb_raw

                total_wb = sum(lane_sums.values()) or 1.0

                for lane_idx, wb_raw in lane_sums.items():
                    pon_raw = ponceau_raw.get(lane_idx, 0.0)
                    if use_ponceau and pon_raw > 0:
                        ratio = wb_raw / pon_raw
                    else:
                        ratio = wb_raw / total_wb

                    records.append(
                        {
                            "slot_index": lane_idx,  # Map to lane for chart color
                            "lane": lane_idx,
                            "label": f"Lane {lane_idx + 1} (Sum)",
                            "wb_band_position": 0,
                            "wb_raw": round(wb_raw, 4),
                            "percent_of_total": round((wb_raw / total_wb) * 100, 2),
                            "ponceau_raw": round(pon_raw, 4) if use_ponceau else None,
                            "ratio": round(ratio, 6),
                            "normalised_ratio": ratio,
                            "is_ladder": lane_types.get(lane_idx, "Sample") == "Ladder",
                            "is_summed": True,  # Flag for the text renderer
                        }
                    )
            else:
                # Standard per-slot multiplexing logic
                total_wb = (
                    sum(
                        float(b.integrated_intensity)
                        if float(b.integrated_intensity) > 1e-6
                        else float(b.peak_height)
                        for _, b in bands_to_analyze
                    )
                    or 1.0
                )

                for slot_idx, b in bands_to_analyze:
                    lane_idx = b.lane_index
                    wb_raw = float(b.integrated_intensity)
                    if wb_raw < 1e-6:
                        wb_raw = float(b.peak_height)

                    pon_raw = ponceau_raw.get(lane_idx, 0.0)

                    if use_ponceau and pon_raw > 0:
                        ratio = wb_raw / pon_raw
                    else:
                        ratio = wb_raw / total_wb

                    label = (
                        f"S{slot_idx + 1} (L{lane_idx + 1})"
                        if has_user_slots
                        else f"Lane {lane_idx + 1}"
                    )

                    records.append(
                        {
                            "slot_index": slot_idx,
                            "lane": lane_idx,
                            "label": label,
                            "wb_band_position": int(b.position),
                            "wb_raw": round(wb_raw, 4),
                            "percent_of_total": round((wb_raw / total_wb) * 100, 2),
                            "ponceau_raw": round(pon_raw, 4) if use_ponceau else None,
                            "ratio": round(ratio, 6),
                            "normalised_ratio": ratio,
                            "is_ladder": lane_types.get(lane_idx, "Sample") == "Ladder",
                            "is_summed": False,  # Flag for the text renderer
                        }
                    )

            df = pd.DataFrame(records)

            # ── Step 4: Optionally scale control lane to 1.0 ──
            if self.chk_normalize_one.isChecked() and not df.empty:
                sample_rows = df[~df["is_ladder"]]
                if not sample_rows.empty:
                    ref_idx = self.combo_ref_lane.currentIndex()
                    if ref_idx > 0:
                        ctrl_lane = ref_idx - 1
                        ctrl_rows = df[df["lane"] == ctrl_lane]
                    else:
                        ctrl_rows = sample_rows.head(1)

                    if not ctrl_rows.empty:
                        ctrl_ratio = float(ctrl_rows.iloc[0]["ratio"])
                        if ctrl_ratio > 0:
                            df["normalised_ratio"] = df["ratio"] / ctrl_ratio

            if not df.empty and "normalised_ratio" in df.columns:
                df["normalised_ratio"] = df["normalised_ratio"].round(4)

            self._panel.status_message.emit(
                f"Results computed: {len(df)} bands analysed"
                + (" (Ponceau-normalised)" if use_ponceau else "")
            )
            self._panel.results_ready.emit(df)

        except Exception as e:
            self._panel.status_message.emit(f"Error computing results: {e}")
            logger.exception("Densitometry error")

    def _get_lane_types(self) -> dict[int, str]:
        from .wb_lanes import WBLanesStep

        for step in self._panel._steps:
            if isinstance(step, WBLanesStep):
                return step.get_lane_types()
        return {}

    def _on_save_workflow(self, panel: WizardPanel):
        """Pops the dialog, grabs the integrated payload, and saves the file."""
        # 1. Pop the dialog from the shared UI folder
        dialog = SaveWorkflowDialog(panel)
        if not dialog.exec():
            return

        metadata = dialog.get_metadata()

        try:
            # 2. Get the main Panel (the aggregator we modified earlier)
            # WBResultsStep -> WizardPanel -> QStackedWidget -> WesternBlotPanel
            wb_panel = panel.parent().parent()

            # 3. Grab the dual-integrated payload (WB + Ponceau)
            if hasattr(wb_panel, "export_workflow"):
                payload = wb_panel.export_workflow()
            else:
                raise AttributeError("WesternBlotPanel does not have export_workflow method.")

            # 4. Access ProjectManager via the window
            # Adjust 'project_manager' string if your MainWindow variable name differs
            main_window = wb_panel.window()
            if not hasattr(main_window, "project_manager"):
                QMessageBox.critical(panel, "Error", "Project Manager not found in main window.")
                return

            active_pm = main_window.project_manager

            # 5. Save to the /workflows directory
            # Use the panel's own plugin_id rather than window().current_module_id —
            # the isolated-mode daemon window (ClosableMainWindow) has no such
            # attribute, only the Hub's own in-process WorkspaceWindow does.
            actual_module_id = wb_panel.plugin_id

            filename = active_pm.save_workflow(
                module_id=actual_module_id, payload=payload, metadata=metadata
            )

            QMessageBox.information(panel, "Success", f"Workflow saved successfully:\n{filename}")

        except Exception as e:
            logger.error(f"Workflow save failed: {e}")
            QMessageBox.critical(panel, "Save Error", f"Failed to save workflow:\n{str(e)}")
