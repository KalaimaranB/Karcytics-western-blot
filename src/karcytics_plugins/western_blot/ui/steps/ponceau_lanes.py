"""Ponceau Stain — Step 2: Lane Detection & Mapping."""

from __future__ import annotations

import logging

from karcytics_sdk.plugin import WizardPanel
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .base_lanes_step import BaseLanesStep

logger = logging.getLogger(__name__)


class PonceauLanesStep(BaseLanesStep):
    """Detect Ponceau lanes and map them to WB lanes."""

    label = "Pon. Lanes"

    _step_title = "Step 2: Ponceau Lanes"
    _step_subtitle = "Detect Ponceau lanes and map them to the Western Blot lanes."
    _detect_btn_text = "🔍  Detect Ponceau Lanes"

    def get_analyzer(self, panel: WizardPanel):
        return panel.ponceau_analyzer

    def _build_extra_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        self.lbl_mismatch = QLabel("")
        self.lbl_mismatch.setWordWrap(True)
        layout.addWidget(self.lbl_mismatch)

        self._mapping_group = QGroupBox("Lane Mapping  —  Ponceau lane → WB lane")
        mapping_top = QVBoxLayout(self._mapping_group)

        map_hint = QLabel("Select the corresponding WB lane for each Ponceau lane.")
        map_hint.setWordWrap(True)
        map_hint.setObjectName("subtitle")
        mapping_top.addWidget(map_hint)

        self._mapping_container = QVBoxLayout()
        mapping_top.addLayout(self._mapping_container)
        self._mapping_combos: list[QComboBox] = []

        self.lbl_no_lanes = QLabel("(detect lanes first)")
        self.lbl_no_lanes.setObjectName("subtitle")
        mapping_top.addWidget(self.lbl_no_lanes)

        layout.addWidget(self._mapping_group)
        self._wb_lane_count = 0

    def _on_enter_hook(self) -> None:
        from .wb_lanes import WBLanesStep

        for step in self._panel._steps:
            if isinstance(step, WBLanesStep):
                self._wb_lane_count = len(self._panel.analyzer.state.lanes)
                break

    def _on_lanes_updated(self, num_lanes: int) -> None:
        self._check_lane_count_match(num_lanes)
        self._rebuild_mapping(num_lanes)

    def _check_lane_count_match(self, pon_count: int) -> None:
        wb_count = self._wb_lane_count
        if wb_count == 0:
            self.lbl_mismatch.setText("")
            return
        if pon_count == wb_count:
            self.lbl_mismatch.setText(
                f"✅  Ponceau lanes ({pon_count}) match WB lanes ({wb_count})."
            )
            self.lbl_mismatch.setStyleSheet(f"color: {Colors.SUCCESS};")
        else:
            self.lbl_mismatch.setText(
                f"⚠️  Ponceau has {pon_count} lanes but WB has {wb_count} lanes.\n"
                f"Use the mapping below to assign which Ponceau lane corresponds "
                f"to each WB lane.  Set extras to 'Skip'."
            )
            self.lbl_mismatch.setStyleSheet(f"color: {Colors.ACCENT_WARNING};")

    def _rebuild_mapping(self, pon_lane_count: int) -> None:
        for combo in self._mapping_combos:
            combo.setParent(None)
            combo.deleteLater()
        self._mapping_combos.clear()
        while self._mapping_container.count():
            item = self._mapping_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.lbl_no_lanes.setVisible(False)
        wb_count = max(self._wb_lane_count, pon_lane_count)

        for pon_idx in range(pon_lane_count):
            row = QHBoxLayout()
            lbl = QLabel(f"Ponceau lane {pon_idx + 1}  →")
            lbl.setFixedWidth(130)
            combo = QComboBox()

            # 1. Add Skip (Index 0)
            combo.addItem("Skip (no WB match)")

            # 2. Add the WB Lanes using wb_count! (Indices 1 to N)
            options = [f"WB Lane {i + 1}" for i in range(wb_count)]
            combo.addItems(options)

            # 3. Add the ignore types (Indices N+1, N+2)
            combo.addItems(["Ladder", "Exclude"])

            mapping = self.get_analyzer(self._panel).lane_mapping
            if mapping and pon_idx in mapping:
                combo.setCurrentIndex(mapping[pon_idx] + 1)
            else:
                combo.setCurrentIndex(pon_idx + 1 if pon_idx < wb_count else 0)

            combo.currentIndexChanged.connect(lambda _, p=self._panel: self._save_state(p))
            self._mapping_combos.append(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            wrapper = QWidget()
            wrapper.setLayout(row)
            self._mapping_container.addWidget(wrapper)

    def _save_state(self, panel: WizardPanel) -> None:
        mapping: dict[int, int] = {}
        for pon_idx, combo in enumerate(self._mapping_combos):
            ci = combo.currentIndex()
            if ci == 0:
                continue
            wb_idx = ci - 1
            mapping[pon_idx] = wb_idx
        panel.ponceau_analyzer.lane_mapping = mapping
        logger.info("Ponceau lane mapping: %s", mapping)

        if hasattr(self, "_panel") and self._panel:
            self._panel.state_changed.emit()

    def _get_status_prefix(self) -> str:
        return "Ponceau: "
