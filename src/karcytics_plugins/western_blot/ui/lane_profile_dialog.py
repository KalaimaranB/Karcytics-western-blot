"""Dialog for viewing lane density profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..analysis.state import AnalysisState

if TYPE_CHECKING:
    from matplotlib.widgets import SpanSelector


class LaneProfileDialog(QDialog):
    """Displays the densitometry profile for each lane.

    Shows the oriented display profile (bands always appear as positive
    peaks), estimated baseline, and detected band positions.

    The user can:
    - Left-click to snap to the nearest peak and add a band.
    - Drag to select a region and add it as a band.
    - Right-click on a band marker (▲) to remove it.

    Signals:
        profile_hovered(lane_idx, y_pos): Mouse moved over plot.
        profile_clicked(lane_idx, y_pos): Left-click on plot.
        profile_range_selected(lane_idx, y_start, y_end): Drag selection.
        profile_band_removed(lane_idx, y_pos): Right-click on marker.
    """

    profile_hovered = Signal(int, float)
    profile_clicked = Signal(int, float, bool)
    profile_range_selected = Signal(int, float, float, bool)
    profile_band_removed = Signal(int, float)

    def __init__(self, state: AnalysisState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Lane Profile Viewer")
        self.setMinimumSize(720, 520)
        self.resize(960, 680)

        self._drag_active = False

        self._setup_ui()
        self._populate_lanes()
        self._update_plot()

    # ── UI construction ───────────────────────────────────────────────

    def _setup_ui(self) -> None:
        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
        )
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
        from matplotlib.figure import Figure

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Controls row
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Lane:"))

        self.combo_lane = QComboBox()
        self.combo_lane.setMinimumWidth(100)
        self.combo_lane.currentIndexChanged.connect(self._on_lane_changed)
        controls.addWidget(self.combo_lane)
        controls.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        controls.addWidget(self.btn_close)
        layout.addLayout(controls)

        # Hint — word-wrap so it never clips on small windows
        hint = QLabel(
            "Drag on the plot to manually add a band region.  "
            "Left-click to snap to the nearest peak.  "
            "Right-click on a marker (▲) to remove that band."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b949e; font-style: italic; padding: 2px 0;")
        layout.addWidget(hint)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.span_selector: SpanSelector | None = None

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_mouse_leave)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    # ── Lane population ───────────────────────────────────────────────

    def _populate_lanes(self) -> None:
        self.combo_lane.clear()
        if not self.state.lanes:
            self.combo_lane.addItem("No lanes detected")
            self.combo_lane.setEnabled(False)
            return
        self.combo_lane.setEnabled(True)
        for i in range(len(self.state.lanes)):
            self.combo_lane.addItem(f"Lane {i + 1}")

    def _on_lane_changed(self, idx: int) -> None:
        if idx >= 0:
            self._update_plot()

    # ── Plot ──────────────────────────────────────────────────────────

    def _update_plot(self) -> None:
        """Redraw the plot for the currently selected lane."""
        import numpy as np
        from matplotlib.widgets import SpanSelector

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        idx = self.combo_lane.currentIndex()

        if not self.state.profiles or idx < 0 or idx >= len(self.state.profiles):
            ax.text(
                0.5,
                0.5,
                "No profile data available.\nRun 'Detect Bands' first.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # state.profiles stores the oriented display_profile — bands are
        # always positive peaks regardless of original image polarity.
        display_profile = np.asarray(self.state.profiles[idx], dtype=np.float64)
        baseline = (
            np.asarray(self.state.baselines[idx], dtype=np.float64)
            if self.state.baselines
            else None
        )

        x = np.arange(len(display_profile))
        ax.plot(x, display_profile, label="Density Profile", color="#2c3e50", linewidth=1.5)

        if baseline is not None:
            ax.plot(
                x,
                baseline,
                label="Estimated Baseline",
                color="#e74c3c",
                linestyle="--",
                linewidth=1.5,
            )
            ax.fill_between(
                x,
                baseline,
                display_profile,
                where=(display_profile > baseline).tolist(),
                color="#3498db",
                alpha=0.3,
                label="Band Area",
            )

        # Band markers — positions are in display_profile coordinates
        lane_bands = [b for b in self.state.bands if b.lane_index == idx]
        for b in lane_bands:
            pos = int(b.position)
            y_val = display_profile[pos] if 0 <= pos < len(display_profile) else b.raw_height
            ax.plot(
                pos,
                y_val,
                marker="^",
                color="#f39c12",
                markersize=9,
                zorder=5,
                markeredgecolor="#c0392b",
                markeredgewidth=0.8,
            )
            if b.width > 0:
                half_w = b.width / 2.0
                ax.axvspan(
                    max(0, b.position - half_w),
                    min(len(display_profile) - 1, b.position + half_w),
                    color="#f1c40f",
                    alpha=0.2,
                )

        ax.set_title(f"Density Profile — Lane {idx + 1}")
        ax.set_xlabel("Vertical Position (pixels)")
        ax.set_ylabel("Intensity")
        # Pixel 0 is at the top of the gel image
        ax.set_xlim(len(display_profile) - 1, 0)
        ax.grid(True, linestyle=":", alpha=0.6)
        if baseline is not None or lane_bands:
            ax.legend(fontsize=8)

        self.figure.tight_layout()

        # Re-initialise SpanSelector for the new axes instance
        self.span_selector = SpanSelector(
            ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="#f1c40f"),
            interactive=False,
            drag_from_anywhere=False,
        )
        self._drag_active = False

        # --- NEW: Initialize the tracking line ---
        self._hover_line = ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        self._hover_line.set_visible(False)
        # -----------------------------------------
        self.canvas.draw()

    # ── Mouse events ──────────────────────────────────────────────────

    def _on_mouse_move(self, event) -> None:
        if getattr(self, "combo_lane", None) is None:
            return

        idx = self.combo_lane.currentIndex()
        if idx < 0:
            return

        if event.button is not None and event.button != 0:
            self._drag_active = True

        if not event.inaxes:
            self._on_mouse_leave(event)
            return

        # --- NEW: Move the Red Line! ---
        if hasattr(self, "_hover_line") and self._hover_line:
            self._hover_line.set_xdata([event.xdata, event.xdata])
            self._hover_line.set_visible(True)
            self.canvas.draw_idle()
        # -------------------------------

        self.profile_hovered.emit(idx, float(event.xdata))

    def _on_mouse_leave(self, event) -> None:
        if getattr(self, "combo_lane", None) is None:
            return

        # --- NEW: Hide the Red Line! ---
        if hasattr(self, "_hover_line") and self._hover_line:
            self._hover_line.set_visible(False)
            self.canvas.draw_idle()
        # -------------------------------

        idx = self.combo_lane.currentIndex()
        if idx >= 0:
            self.profile_hovered.emit(idx, -1.0)

    def _on_mouse_release(self, event) -> None:
        if not event.inaxes or self.combo_lane.currentIndex() < 0:
            self._drag_active = False
            return

        if self._drag_active:
            self._drag_active = False
            return

        idx = self.combo_lane.currentIndex()

        if event.button == 1:
            try:
                mode = getattr(self.toolbar.mode, "name", self.toolbar.mode)
                if str(mode) not in ("", "NONE"):
                    return
            except Exception:
                pass

            # --- NEW: Check for Ctrl / Cmd key ---
            modifiers = QGuiApplication.keyboardModifiers()
            # ControlModifier handles 'Cmd' on Mac and 'Ctrl' on Windows. MetaModifier handles the literal 'Ctrl' on Mac.
            is_ctrl = bool(
                modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            )
            auto_snap = not is_ctrl  # If Ctrl is held, do NOT auto-snap

            self.profile_clicked.emit(idx, float(event.xdata), auto_snap)

        elif event.button == 3:
            self.profile_band_removed.emit(idx, float(event.xdata))

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        self._drag_active = True
        idx = self.combo_lane.currentIndex()
        if idx < 0:
            return

        lo, hi = min(xmin, xmax), max(xmin, xmax)
        if hi - lo < 3:
            return

        # --- NEW: Check for Ctrl / Cmd key during drag ---
        modifiers = QGuiApplication.keyboardModifiers()
        is_ctrl = bool(
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        auto_snap = not is_ctrl

        self.profile_range_selected.emit(idx, float(lo), float(hi), auto_snap)
