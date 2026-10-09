"""Tests that the plugin's own widgets follow a live Hub theme switch.

Mirrors what `ui_daemon_runtime`'s `theme_changed` handler does in the real
isolated process: update `DynamicColors`, re-apply tracked styles, emit
`theme_manager.theme_changed`.
"""

import re

import pytest
from karcytics_sdk.plugin.theme_fallback import DynamicColors, theme_manager
from matplotlib.colors import to_hex
from PySide6.QtWidgets import QWidget

from karcytics_plugins.western_blot.analysis.state import AnalysisState
from karcytics_plugins.western_blot.ui.image_canvas import ImageCanvas
from karcytics_plugins.western_blot.ui.lane_profile_dialog import LaneProfileDialog
from karcytics_plugins.western_blot.ui.steps.ponceau_bands import _FactorChart
from karcytics_plugins.western_blot.ui.western_blot_panel import WesternBlotPanel

pytestmark = pytest.mark.ui


def _push_theme(name: str) -> None:
    DynamicColors.set_theme(name)
    theme_manager._apply_dynamic_styles()
    theme_manager.theme_changed.emit()


@pytest.fixture(autouse=True)
def _dark_theme():
    _push_theme("dark")
    yield
    _push_theme("dark")


class TestThemeFollowing:
    def test_image_canvas_background_follows_theme(self, qtbot):
        canvas = ImageCanvas()
        qtbot.addWidget(canvas)
        assert DynamicColors.DARK["BG_DARKEST"] in canvas.styleSheet()

        _push_theme("light")

        assert DynamicColors.LIGHT["BG_DARKEST"] in canvas.styleSheet()
        assert DynamicColors.DARK["BG_DARKEST"] not in canvas.styleSheet()

    def test_factor_chart_redraws_in_new_theme(self, qtbot):
        chart = _FactorChart()
        qtbot.addWidget(chart)
        chart.plot_factors({0: 1.0, 1: 1.6}, 2, label_prefix="Pon.")
        assert to_hex(chart.fig.get_facecolor()) == DynamicColors.DARK["BG_DARK"]

        _push_theme("light")

        assert to_hex(chart.fig.get_facecolor()) == DynamicColors.LIGHT["BG_DARK"]
        assert to_hex(chart.ax.get_facecolor()) == DynamicColors.LIGHT["BG_DARK"]
        assert len(chart.ax.patches) == 2

    def test_factor_chart_ignores_theme_change_before_first_plot(self, qtbot):
        chart = _FactorChart()
        qtbot.addWidget(chart)

        _push_theme("light")

        assert chart.canvas is None

    def test_lane_profile_plot_follows_theme(self, qtbot):
        dialog = LaneProfileDialog(AnalysisState())
        qtbot.addWidget(dialog)
        assert to_hex(dialog.figure.get_facecolor()) == DynamicColors.DARK["BG_DARK"]

        _push_theme("light")

        assert to_hex(dialog.figure.get_facecolor()) == DynamicColors.LIGHT["BG_DARK"]

    def test_theme_switch_leaves_no_unresolved_placeholders(self, qtbot):
        """A `{KEY}` the palette doesn't know stays literal and silently breaks the rule."""
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.data_ready, timeout=2000):
            panel.begin_async_init()
        panel._setup_screen.analysis_requested.emit(True)

        _push_theme("light")

        for widget in panel.findChildren(QWidget):
            assert not re.search(r"\{[A-Z_]+\}", widget.styleSheet()), widget
