"""Western Blot — Step 2: Lane Detection."""

from __future__ import annotations

import logging

from karcytics_sdk.plugin import WizardPanel
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .base_lanes_step import BaseLanesStep

logger = logging.getLogger(__name__)


class WBLanesStep(BaseLanesStep):
    """Detect lane boundaries in the preprocessed image."""

    label = "Lanes"

    _step_title = "Step 2: Detect Lanes"
    _step_subtitle = "Auto-detect or manually set the number of lanes."

    def get_analyzer(self, panel: WizardPanel):
        return panel.analyzer

    def _build_extra_ui(self, layout: QVBoxLayout, panel: WizardPanel) -> None:
        self._lane_type_group = QGroupBox("Lane Types")
        lane_type_layout = QVBoxLayout(self._lane_type_group)

        desc = QLabel("Mark lanes as Ladder or Exclude to skip them in analysis:")
        desc.setWordWrap(True)
        lane_type_layout.addWidget(desc)

        self._lane_type_container = QVBoxLayout()
        self.lane_type_combos: list[QComboBox] = []
        lane_type_layout.addLayout(self._lane_type_container)

        self.lbl_no_lanes = QLabel("(detect lanes first)")
        self.lbl_no_lanes.setObjectName("subtitle")
        lane_type_layout.addWidget(self.lbl_no_lanes)

        layout.addWidget(self._lane_type_group)

    def _on_lanes_updated(self, num_lanes: int) -> None:
        for combo in self.lane_type_combos:
            combo.setParent(None)
            combo.deleteLater()
        self.lane_type_combos.clear()
        self.lbl_no_lanes.setVisible(False)

        while self._lane_type_container.count():
            item = self._lane_type_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i in range(num_lanes):
            row = QHBoxLayout()
            lbl = QLabel(f"Lane {i + 1}:")
            lbl.setFixedWidth(56)
            combo = QComboBox()
            combo.addItems(["Sample", "Ladder", "Exclude"])
            combo.setToolTip(
                "Sample: include in analysis\nLadder: molecular weight marker\nExclude: skip entirely"
            )

            # Select correct default if lanes exist
            lanes = self.get_analyzer(self._panel).state.lanes
            if lanes and i < len(lanes):
                idx = combo.findText(lanes[i].lane_type)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

            combo.currentIndexChanged.connect(
                lambda _, idx=i, cb=combo: self._update_lane_type(idx, cb.currentText())
            )
            self.lane_type_combos.append(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            wrapper = QWidget()
            wrapper.setLayout(row)
            self._lane_type_container.addWidget(wrapper)

    def _update_lane_type(self, lane_idx: int, lane_type: str) -> None:
        lanes = self.get_analyzer(self._panel).state.lanes
        if lanes and lane_idx < len(lanes):
            lanes[lane_idx].lane_type = lane_type
            self._panel.state_changed.emit()
            logger.debug(f"Lane {lane_idx} marked as {lane_type}")

    def get_lane_types(self) -> dict[int, str]:
        return {i: combo.currentText() for i, combo in enumerate(self.lane_type_combos)}

    def _get_default_lane_type(self, index: int) -> str:
        return self.get_lane_types().get(index, "Sample")
