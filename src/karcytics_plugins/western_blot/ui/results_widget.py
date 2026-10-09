"""Results display widget — chart, slot-based band comparison, popout table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from karcytics_sdk.plugin import BioCaptionLabel, BioSpinBox, BioTableWidget, SecondaryButton
from karcytics_sdk.plugin.theme_fallback import Colors, theme_manager
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ── Data table popout ─────────────────────────────────────────────────────────


class DataTableDialog(QDialog):
    """Full results table in a scrollable, resizable popout window."""

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Full Results Table")
        self.resize(920, 520)
        theme_manager.apply_style(self, "background: {BG_DARK};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        table = BioTableWidget()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

        col_labels = {
            "lane": "Lane",
            "band": "Band",
            "matched_band": "Matched",
            "position": "Position (px)",
            "raw_intensity": "Raw Intensity",
            "percent_of_total": "% of Total",
            "normalized": "Normalised",
            "ponceau_factor": "Ponceau Factor",
            "ponceau_normalized": "Ponceau Norm.",
            "snr": "SNR",
            "width": "Width (px)",
            "wb_band_position": "Position (px)",
            "wb_raw": "WB Raw Intensity",
            "ponceau_raw": "Ponceau Raw",
            "ratio": "WB/Pon Ratio",
            "normalised_ratio": "Normalised",
        }

        core = ["lane", "band", "position", "raw_intensity", "percent_of_total", "normalized"]
        prof_cols = ["wb_band_position", "wb_raw", "ponceau_raw", "ratio", "normalised_ratio"]
        pon_cols = ["ponceau_factor", "ponceau_normalized"]
        extra = ["snr", "width"]

        cols = [c for c in core + prof_cols + pon_cols + extra if c in df.columns]

        table.setColumnCount(len(cols))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels([col_labels.get(c, c) for c in cols])

        from PySide6.QtGui import QColor as _QColor

        fg_primary = _QColor(Colors.FG_PRIMARY)
        accent_primary = _QColor(Colors.ACCENT_PRIMARY)

        for ri, (_, row) in enumerate(df.iterrows()):
            import numpy as np

            for ci, col in enumerate(cols):
                val = row[col]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    text = ""
                elif col == "lane":
                    text = str(int(val) + 1)
                elif isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

                # ── THE FIX: Force the text color so it never clashes ──
                if col in pon_cols:
                    item.setForeground(accent_primary)
                else:
                    item.setForeground(fg_primary)
                # ───────────────────────────────────────────────────────

                table.setItem(ri, ci, item)

        layout.addWidget(table)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        layout.addWidget(btns)


# ── Chart ─────────────────────────────────────────────────────────────────────


class DensityChart(QWidget):
    """Matplotlib bar chart — WB or Ponceau-corrected values."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.fig = None
        self.canvas = None
        self.axes = None
        theme_manager.apply_style(self, "background-color: {BG_DARK};")

    def _ensure_canvas(self):
        """Lazy loader for Matplotlib components."""
        if self.canvas is not None:
            return

        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(6, 3.2), dpi=100)
        self.fig.patch.set_facecolor(Colors.BG_DARK)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.layout().addWidget(self.canvas)

    def plot_densities(
        self,
        df: pd.DataFrame,
        mode: str = "wb",
        highlighted_lanes: list[int] | None = None,
    ) -> None:
        self._ensure_canvas()
        self.axes.clear()
        self.axes.set_facecolor(Colors.BG_DARK)
        self.axes.tick_params(colors=Colors.FG_SECONDARY)
        for sp in ("top", "right"):
            self.axes.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            self.axes.spines[sp].set_color(Colors.BORDER)

        if df.empty:
            self.axes.text(
                0.5,
                0.5,
                "No data to display",
                ha="center",
                va="center",
                color=Colors.FG_SECONDARY,
                fontsize=13,
                transform=self.axes.transAxes,
            )
            self.canvas.draw()
            return

        use_pon = mode == "ponceau" and "ponceau_normalized" in df.columns
        vcol = "ponceau_normalized" if use_pon else "normalized"

        sample_df = df[~df["is_ladder"]] if "is_ladder" in df.columns else df
        primary = (
            sample_df.sort_values("raw_intensity", ascending=False)
            .groupby("lane", as_index=False)
            .first()
            .sort_values("lane")
        )

        hl = set(highlighted_lanes or [])
        lanes = primary["lane"].values
        vals = primary[vcol].values
        labels = [f"Lane {int(lane) + 1}" for lane in lanes]
        colors = [
            Colors.ACCENT_WARNING
            if int(lane) in hl
            else Colors.CHART_COLORS[int(lane) % len(Colors.CHART_COLORS)]
            for lane in lanes
        ]

        bars = self.axes.bar(
            range(len(lanes)),
            vals,
            color=colors,
            edgecolor="none",
            width=0.6,
            alpha=0.9,
        )
        self.axes.set_xticks(range(len(lanes)))
        self.axes.set_xticklabels(labels, fontsize=9, color=Colors.FG_SECONDARY)

        max_val = float(max(vals)) if len(vals) else 1.0
        for bar, val in zip(bars, vals, strict=False):
            if val > 0:
                self.axes.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=Colors.FG_SECONDARY,
                )

        ylabel = "Ponceau-normalised Density" if use_pon else "Relative Density"
        self.axes.set_ylabel(ylabel, fontsize=10, color=Colors.FG_PRIMARY)
        title = (
            "Band Density — Ponceau Loading Correction Applied"
            if use_pon
            else "Band Density Comparison"
        )
        self.axes.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            color=Colors.FG_PRIMARY,
            pad=8,
        )
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_professor(
        self,
        df: pd.DataFrame,
        has_ponceau: bool = False,
        highlighted_lanes: list[int] | None = None,
        slot_colors: dict[int, str] | None = None,
    ) -> None:
        self._ensure_canvas()
        self.axes.clear()
        self.axes.set_facecolor(Colors.BG_DARK)
        self.axes.tick_params(colors=Colors.FG_SECONDARY)
        for sp in ("top", "right"):
            self.axes.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            self.axes.spines[sp].set_color(Colors.BORDER)

        if df.empty or "normalised_ratio" not in df.columns:
            self.axes.text(
                0.5,
                0.5,
                "No data yet",
                ha="center",
                va="center",
                color=Colors.FG_SECONDARY,
                fontsize=12,
                transform=self.axes.transAxes,
            )
            self.canvas.draw()
            return

        sample = df[~df["is_ladder"]] if "is_ladder" in df.columns else df
        sc = slot_colors or {}

        # The Fix: Read by Slot Index instead of Lane
        if "slot_index" in sample.columns:
            indices = sample["slot_index"].values
            ratios = sample["normalised_ratio"].values
            labels = sample["label"].values
            colors = [
                sc.get(int(idx), Colors.CHART_COLORS[int(idx) % len(Colors.CHART_COLORS)])
                for idx in indices
            ]
        else:
            indices = sample["lane"].values
            ratios = sample["normalised_ratio"].values
            labels = [f"Lane {int(lane) + 1}" for lane in indices]
            colors = [
                sc.get(int(lane), Colors.CHART_COLORS[int(lane) % len(Colors.CHART_COLORS)])
                for lane in indices
            ]

        bars = self.axes.bar(
            range(len(indices)), ratios, color=colors, edgecolor="none", width=0.6, alpha=0.92
        )
        self.axes.set_xticks(range(len(indices)))
        self.axes.set_xticklabels(labels, fontsize=9, color=Colors.FG_SECONDARY)

        max_v = float(max(ratios)) if len(ratios) else 1.0
        for bar, val, c in zip(bars, ratios, colors, strict=False):
            self.axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=c,
            )

        title = "WB / Ponceau Ratio (normalised)" if has_ponceau else "WB Band Density (normalised)"
        self.axes.set_title(title, fontsize=12, fontweight="bold", color=Colors.FG_PRIMARY, pad=8)
        self.axes.set_ylabel(
            "Ratio" if has_ponceau else "Density", fontsize=10, color=Colors.FG_PRIMARY
        )

        self.fig.tight_layout()
        self.canvas.draw()

    def save_chart(self, path: Path) -> None:
        self.fig.savefig(
            str(path),
            dpi=300,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
        )


# ── Results widget ────────────────────────────────────────────────────────────


class ResultsWidget(QWidget):
    """Chart + slot-based band comparison + popout full data table."""

    _SLOT_COLORS = [
        "#f85149",  # red
        "#58a6ff",  # blue
        "#3fb950",  # green
        "#d29922",  # amber
        "#a371f7",  # purple
        "#f778ba",  # pink
        "#2dccb8",  # teal
        "#79c0ff",  # light blue
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._chart_mode = "wb"
        self._canvas_ref = None
        self._slots: list = []
        self._active_slot: int | None = None
        self._num_slots = 2
        self._slot_btns: list[QPushButton] = []
        self._setup_ui()
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def set_canvas(self, canvas) -> None:
        self._canvas_ref = canvas

    # ── UI ────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Chart
        self.chart = DensityChart(self)
        layout.addWidget(self.chart, stretch=3)

        # WB / Ponceau toggle
        self._toggle_row = QWidget()
        tr = QHBoxLayout(self._toggle_row)
        tr.setContentsMargins(4, 0, 4, 0)
        tr.setSpacing(8)
        tr.addWidget(QLabel("Show:"))
        self._btn_wb = QRadioButton("WB Normalised")
        self._btn_pon = QRadioButton("Ponceau Corrected")
        self._btn_wb.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._btn_wb, 0)
        self._mode_grp.addButton(self._btn_pon, 1)
        self._mode_grp.idClicked.connect(self._on_mode_changed)
        tr.addWidget(self._btn_wb)
        tr.addWidget(self._btn_pon)
        tr.addStretch()
        self._toggle_row.setVisible(False)
        layout.addWidget(self._toggle_row)

        # Export row
        exp = QHBoxLayout()
        exp.setSpacing(4)
        self.btn_table = SecondaryButton("📋 Full Table")
        self.btn_csv = SecondaryButton("📄 CSV")
        self.btn_excel = SecondaryButton("📊 Excel")
        self.btn_png = SecondaryButton("🖼️ Chart")
        for b in (self.btn_table, self.btn_csv, self.btn_excel, self.btn_png):
            b.setMinimumHeight(28)
            exp.addWidget(b)
        self.btn_table.clicked.connect(self._show_table)
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_excel.clicked.connect(self._export_excel)
        self.btn_png.clicked.connect(self._export_png)
        layout.addLayout(exp)

        # Band comparison group
        cg = QGroupBox("Band Comparison — Fold Change")
        cg_l = QVBoxLayout(cg)
        cg_l.setSpacing(5)

        # Slot count
        cr = QHBoxLayout()
        cr.addWidget(QLabel("Bands to compare:"))
        self._spin_slots = BioSpinBox()
        self._spin_slots.setRange(2, 12)
        self._spin_slots.setValue(2)
        self._spin_slots.setToolTip(
            "Number of bands to compare.\n"
            "Click a Band button to enter selection mode,\n"
            "then click a band on the image."
        )
        self._spin_slots.valueChanged.connect(self._on_slots_changed)
        # valueChanged is wired externally by WesternBlotPanel.set_results_widget
        # so that changing the count also triggers a results recompute.
        cr.addWidget(self._spin_slots)
        cr.addStretch()
        cg_l.addLayout(cr)

        # Instructions
        hint = BioCaptionLabel(
            "1. Click any band on the image to add or remove it.\n"
            "2. Click a colored band button below to clear that specific slot."
        )
        cg_l.addWidget(hint)

        # Slot buttons container
        self._slots_container = QWidget()
        self._slots_layout = QVBoxLayout(self._slots_container)
        self._slots_layout.setSpacing(3)
        self._slots_layout.setContentsMargins(0, 0, 0, 0)
        cg_l.addWidget(self._slots_container)

        # Clear button
        btn_clr = QPushButton("✖  Clear All")
        theme_manager.apply_style(
            btn_clr,
            "QPushButton { background: {BG_MEDIUM};"
            " color: {FG_SECONDARY}; border: 1px solid {BORDER};"
            " border-radius: 5px; padding: 3px 10px; }"
            "QPushButton:hover { background: {BG_LIGHT}; }",
        )
        btn_clr.clicked.connect(self._clear_all)
        cg_l.addWidget(btn_clr)

        # Result text
        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.TextFormat.RichText)
        theme_manager.apply_style(self.lbl_result, "color: {FG_PRIMARY}; font-size: 11px;")
        cg_l.addWidget(self.lbl_result)

        layout.addWidget(cg)
        self.comparison_group = cg

        # Build default slots
        self._rebuild_slots(2)

    def _col(self, idx: int) -> str:
        return Colors.CHART_COLORS[idx % len(Colors.CHART_COLORS)]

    def _on_slot_clicked(self, idx: int, checked: bool) -> None:
        if checked:
            self._active_slot = idx
            for i, b in enumerate(self._slot_btns):
                if i != idx:
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
            self._slot_btns[idx].setText(
                f"Band {idx + 1} — 🎯 selecting… click a band on the image"
            )
        else:
            self._active_slot = None
            self._update_slot_label(idx)

    # ── Public API ────────────────────────────────────────────────────

    def set_results(self, df: pd.DataFrame) -> None:
        """Update results data.

        On the **first** call, initialises the slot count from the number
        of lanes.  On subsequent calls, preserves the user's slot count
        and band assignments — only the labels and chart are refreshed.
        """
        is_first = self._df is None
        self._df = df.copy()
        import pandas as pd

        if is_first:
            # First result: set slot count from lane count
            n = max(2, int(df["lane"].nunique())) if not df.empty else 2
            self._spin_slots.blockSignals(True)
            self._spin_slots.setValue(n)
            self._spin_slots.blockSignals(False)
            self._rebuild_slots(n)
        else:
            # Subsequent updates: preserve slot count and assignments.
            # Just refresh labels so ratio values shown in buttons stay current.
            for i in range(self._num_slots):
                self._update_slot_label(i)

        # Detect Ponceau
        if "ponceau_raw" in df.columns:
            pon_vals = pd.to_numeric(df["ponceau_raw"], errors="coerce").fillna(0)
            bool((pon_vals > 0).any())

        self._toggle_row.setVisible(False)
        self._chart_mode = "ratio"
        self._refresh_chart()
        self._update_canvas_markers()
        self._render_result()

    def assign_band_to_active_slot(self, band) -> None:
        """Called when user clicks a band on the canvas while in slot-select mode."""
        idx = self._active_slot
        if idx is None:
            return

        self._slots[idx] = band
        self._slot_btns[idx].blockSignals(True)
        self._slot_btns[idx].setChecked(False)
        self._slot_btns[idx].blockSignals(False)
        self._active_slot = None

        try:
            self._update_slot_label(idx)
            self._update_canvas_markers()
            self._render_result()
            self._refresh_chart()
        except Exception:
            import traceback

            traceback.print_exc()

    # Back-compat shim
    def highlight_band_for_comparison(self, band) -> None:
        self.assign_band_to_active_slot(band)

    def update_pairwise_comparison(self, _bands: list) -> None:
        pass

    # ── Internal ──────────────────────────────────────────────────────
    def _rebuild_slots(self, n: int) -> None:
        """Recreate n slot buttons, preserving existing assignments where possible."""
        old_slots = list(self._slots)
        old_n = self._num_slots

        self._active_slot = None
        self._slots = [None] * n
        self._num_slots = n

        while self._slots_layout.count():
            item = self._slots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._slot_btns = []
        for i in range(n):
            c = self._col(i)
            btn = QPushButton(f"Slot {i + 1} — Empty")
            btn.setMinimumHeight(30)
            theme_manager.apply_style(
                btn,
                f"QPushButton {{ border: 2px solid {c}; border-radius: 5px;"
                f" background: {{BG_DARK}}; color: {{FG_SECONDARY}};"
                f" padding: 4px 8px; text-align: left; font-size: 11px; }}"
                f"QPushButton:hover {{ background: {c}22; color: {c}; font-weight: 600; }}",
            )
            # Clicking the button now acts as a quick-delete for that slot
            btn.clicked.connect(lambda _, idx=i: self._clear_slot(idx))
            self._slot_btns.append(btn)
            self._slots_layout.addWidget(btn)

        # Restore previous assignments
        for i in range(min(n, old_n)):
            if i < len(old_slots) and old_slots[i] is not None:
                self._slots[i] = old_slots[i]
                self._update_slot_label(i)

        self.lbl_result.setText("")
        self._update_canvas_markers()

    def _clear_slot(self, idx: int) -> None:
        """Removes a band, compacts the list, and auto-shrinks the UI if possible."""
        if self._slots[idx] is not None:
            self._slots[idx] = None

            # Compact the list by shifting None values to the end
            valid_bands = [b for b in self._slots if b is not None]

            # Calculate the new required number of slots (minimum 2 to show fold-change)
            new_slot_count = max(2, len(valid_bands))

            # Update the underlying data array FIRST
            self._slots = valid_bands + [None] * (
                max(self._num_slots, new_slot_count) - len(valid_bands)
            )

            # ── THE FIX: Auto-shrink the UI spinbox if we have excess empty slots ──
            if new_slot_count < self._num_slots:
                # Setting this value automatically triggers _rebuild_slots() to destroy the extra button!
                self._spin_slots.setValue(new_slot_count)
            else:
                # If we are already at the minimum of 2, just update the text labels
                for i in range(self._num_slots):
                    self._update_slot_label(i)
                self._update_canvas_markers()
                self._refresh_chart()
            # ───────────────────────────────────────────────────────────────────────

            # Force the backend math engine to recompute
            p = self.parent()
            while p:
                if hasattr(p, "_wb_results_step"):
                    p._wb_results_step._compute_results()
                    if hasattr(p, "state_changed"):
                        p.state_changed.emit()
                    break
                p = p.parent()

    def toggle_band_selection(self, band) -> bool:
        """Toggles a band in the slots. Returns True if added, False if removed."""
        # 1. Check if it's already selected (match by exact lane and index)
        existing_idx = -1
        for i, b in enumerate(self._slots):
            if (
                b is not None
                and b.lane_index == band.lane_index
                and b.band_index == band.band_index
            ):
                existing_idx = i
                break

        # 2. If it's there, remove it
        if existing_idx != -1:
            self._clear_slot(existing_idx)
            return False

        # 3. If it's not there, find the first empty slot
        empty_idx = -1
        for i, b in enumerate(self._slots):
            if b is None:
                empty_idx = i
                break

        # 4. Add it in
        if empty_idx != -1:
            self._slots[empty_idx] = band
        # Auto-expand the slots if full (up to a safe UI limit of 12)
        elif self._num_slots < 12:
            new_count = self._num_slots + 1
            self._spin_slots.setValue(new_count)  # This rebuilds slots automatically!
            self._slots[-1] = band
            empty_idx = new_count - 1
        else:
            return False  # Reached max capacity

        self._update_slot_label(empty_idx if empty_idx != -1 else -1)
        self._update_canvas_markers()
        self._render_result()
        self._refresh_chart()
        self._spin_slots.valueChanged.emit(self._num_slots)  # Force math update
        return True

    def _update_slot_label(self, idx: int) -> None:
        if idx == -1:
            return
        band = self._slots[idx]
        if band is None:
            self._slot_btns[idx].setText(f"Slot {idx + 1} — Empty")
            return

        intensity = float(band.integrated_intensity)
        intensity_src = "peak" if intensity < 1e-6 else "raw"
        if intensity < 1e-6:
            intensity = float(band.peak_height)

        norm_txt = f"  ·  {intensity_src} {intensity:.1f}"

        if self._df is not None and not self._df.empty:
            if "slot_index" in self._df.columns:
                mask = self._df["slot_index"] == idx
            else:
                mask = self._df["lane"].astype(int) == int(band.lane_index)

            rows = self._df[mask]
            if not rows.empty:
                ratio = float(rows.iloc[0].get("normalised_ratio", rows.iloc[0].get("ratio", 0.0)))
                norm_txt += f"  ·  norm {ratio:.4f}"

        self._slot_btns[idx].setText(
            f"Band {idx + 1}  ·  Lane {band.lane_index + 1}"
            f"  pos {band.position}px  SNR {band.snr:.1f}"
            f"  raw {band.integrated_intensity:.1f}{norm_txt}"
        )

    def _clear_all(self) -> None:
        self._active_slot = None
        for i in range(self._num_slots):
            self._slots[i] = None
            self._update_slot_label(i)
        self.lbl_result.setText("")
        self._update_canvas_markers()
        self._refresh_chart()
        self._spin_slots.valueChanged.emit(self._num_slots)

    def _update_canvas_markers(self) -> None:
        if self._canvas_ref is None:
            return
        if hasattr(self._canvas_ref, "set_all_comparison_slots"):
            slot_map = {}
            for i, band in enumerate(self._slots):
                if band is not None:
                    slot_map[(band.lane_index, band.band_index)] = self._col(i)
            self._canvas_ref.set_all_comparison_slots(slot_map)
        elif hasattr(self._canvas_ref, "set_band_comparison_slots"):
            a = self._slots[0] if len(self._slots) > 0 else None
            b = self._slots[1] if len(self._slots) > 1 else None
            self._canvas_ref.set_band_comparison_slots(a, b)

    def _refresh_chart(self) -> None:
        if self._df is None:
            return
        import pandas as pd

        slot_colors = {}
        for i, band in enumerate(self._slots):
            if band is not None:
                slot_colors[i] = self._col(i)

        pon_vals = pd.to_numeric(self._df["ponceau_raw"], errors="coerce").fillna(0)
        has_pon = bool((pon_vals > 0).any())
        self.chart.plot_professor(self._df, has_ponceau=has_pon, slot_colors=slot_colors)

    def _get_vals(self, band, slot_idx=None):
        if self._df is None or self._df.empty:
            return None, None

        # ── THE FIX: If summed, fetch the total lane value regardless of slot ──
        if "is_summed" in self._df.columns and self._df["is_summed"].iloc[0]:
            mask = self._df["lane"].astype(int) == int(band.lane_index)
            rows = self._df[mask]
            if not rows.empty:
                row = rows.iloc[0]
                ratio = float(row.get("normalised_ratio", row.get("ratio", 0.0)))
                pon_raw = float(row.get("ponceau_raw", 0.0) or 0.0)
                return ratio, pon_raw
        # ───────────────────────────────────────────────────────────────────────

        # Standard robust lookup by exact slot
        if "slot_index" in self._df.columns and slot_idx is not None:
            mask = self._df["slot_index"] == slot_idx
            rows = self._df[mask]
            if not rows.empty:
                row = rows.iloc[0]
                ratio = float(row.get("normalised_ratio", row.get("ratio", 0.0)))
                pon_raw = float(row.get("ponceau_raw", 0.0) or 0.0)
                return ratio, pon_raw

        # Legacy fallback
        mask = self._df["lane"].astype(int) == int(band.lane_index)
        if "band" in self._df.columns:
            mask &= self._df["band"].astype(int) == int(band.band_index)
        rows = self._df[mask]
        if rows.empty:
            return None, None
        row = rows.iloc[0]
        ratio = float(row.get("normalised_ratio", row.get("normalized", 0.0)))
        pon = float(row.get("ponceau_raw", row.get("ponceau_normalized", ratio)))
        return ratio, pon

    def _render_result(self) -> None:
        filled = [(i, b) for i, b in enumerate(self._slots) if b is not None]
        if len(filled) < 2:
            self.lbl_result.setText(
                f"<span style='color:{Colors.FG_SECONDARY}; font-size:10px;'>"
                f"Fill at least 2 slots to compute comparisons.</span>"
            )
            return

        is_summed = (
            self._df is not None
            and "is_summed" in self._df.columns
            and not self._df["is_summed"].empty
            and self._df["is_summed"].iloc[0]
        )

        lines = []

        # ─── 1. FOLD CHANGE VS CONTROL ──────────────────────────────────
        lines.append(
            f"<span style='color:{Colors.FG_PRIMARY}; font-weight:bold; font-size:12px;'>Fold Change vs Control (Slot 1):</span>"
        )

        # Treat Slot 1 as the universal baseline
        ctrl_idx, ctrl_band = filled[0]
        c_norm, c_pon = self._get_vals(ctrl_band, slot_idx=ctrl_idx)
        c_color = self._col(ctrl_idx)

        if is_summed and self._df is not None:
            # If summed, pull the unique lane totals straight from the dataframe
            for _, row in self._df.iterrows():
                l_idx = int(row["lane"])
                ratio = row.get("normalised_ratio", 0.0)
                # Highlight fold changes that aren't exactly 1.000
                val_str = f"<b>{ratio:.3f}x</b>" if abs(ratio - 1.0) > 0.001 else f"{ratio:.3f}x"
                lines.append(f"&nbsp;&nbsp;Lane {l_idx + 1} Total \u2192 {val_str}")
        else:
            # Standard slot-to-slot comparison against Slot 1
            for i, band in filled[1:]:
                c = self._col(i)
                norm, pon = self._get_vals(band, slot_idx=i)

                fc = "N/A"
                if c_norm and norm and c_norm > 1e-6:
                    ratio = norm / c_norm
                    fc = f"<b>{ratio:.3f}x</b>" if abs(ratio - 1.0) > 0.001 else f"{ratio:.3f}x"

                lines.append(
                    f"&nbsp;&nbsp;<span style='color:{c};'>Slot {i + 1}</span> vs "
                    f"<span style='color:{c_color};'>Slot 1</span> &nbsp;\u2192&nbsp; {fc}"
                )

        lines.append("")  # spacer

        # ─── 2. DATA QUALITY CONTROL (QC) ───────────────────────────────
        lines.append(
            f"<span style='color:{Colors.FG_PRIMARY}; font-weight:bold; font-size:12px;'>Data Quality Assessment:</span>"
        )

        low_snr_slots = []
        for i, band in filled:
            # A Signal-to-Noise Ratio under 3.0 is generally considered unreliable in densitometry
            if band.snr < 3.0:
                low_snr_slots.append(str(i + 1))

        if low_snr_slots:
            warn_color = Colors.ACCENT_WARNING
            lines.append(
                f"&nbsp;&nbsp;<span style='color:{warn_color};'>⚠️ Warning: Slots {', '.join(low_snr_slots)} have low SNR (< 3.0). Signal may be background noise.</span>"
            )
        else:
            success_color = Colors.ACCENT_SUCCESS
            lines.append(
                f"&nbsp;&nbsp;<span style='color:{success_color};'>✅ All selected bands have distinct peaks (SNR > 3.0).</span>"
            )

        self.lbl_result.setText("<br>".join(lines))

    def _on_mode_changed(self, btn_id: int) -> None:
        self._chart_mode = "ponceau" if btn_id == 1 else "wb"
        self._refresh_chart()

    def _show_table(self) -> None:
        if self._df is None:
            return
        DataTableDialog(self._df, parent=self).exec()

    def _export_csv(self) -> None:
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "densitometry_results.csv", "CSV Files (*.csv)"
        )
        if path:
            self._df.to_csv(path, index=False)

    def _export_excel(self) -> None:
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "densitometry_results.xlsx", "Excel Files (*.xlsx)"
        )
        if path:
            self._df.to_excel(path, index=False, sheet_name="Results")

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "density_chart.png", "PNG Files (*.png)"
        )
        if path:
            self.chart.save_chart(Path(path))

    def _on_slots_changed(self, val: int) -> None:
        """Handle changes to the number of comparison slots."""
        self._rebuild_slots(val)
        self._refresh_chart()
        self._render_result()

    def _on_theme_changed(self) -> None:
        """Force Matplotlib and internal styles to redraw when the theme changes."""
        # The figure is built lazily — nothing to recolour before the first plot.
        if self.chart.fig is not None:
            self.chart.fig.patch.set_facecolor(Colors.BG_DARK)
            self.chart.canvas.draw_idle()

        # Force the chart to redraw with the new palette
        self._refresh_chart()

        # Update the text labels to use the new colors
        self._render_result()
