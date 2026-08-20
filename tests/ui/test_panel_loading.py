"""Tests for the panel_ready / data_ready loading-screen protocol.

See karcytics_sdk.plugin.base.PluginBase — the Hub's PluginLoaderManager
duck-types hasattr(panel, "panel_ready") and drives the loader's
"Loading workspace data..." transition off these two signals.
"""

import pytest

from karcytics_plugins.western_blot.ui.western_blot_panel import WesternBlotPanel

pytestmark = pytest.mark.ui


class TestLoadingProtocol:
    def test_panel_exposes_ready_signals(self, qtbot):
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)
        assert hasattr(panel, "panel_ready")
        assert hasattr(panel, "data_ready")

    def test_heavy_widgets_not_built_synchronously(self, qtbot):
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)
        assert panel.canvas is None
        assert panel.results_widget is None

    def test_begin_async_init_emits_panel_ready_then_data_ready(self, qtbot):
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)

        # Both signals must be armed before begin_async_init() runs — data_ready
        # is scheduled via QTimer.singleShot(0, ...) right after panel_ready
        # fires, so it can beat a second, separately-armed waitSignal() call.
        with qtbot.waitSignals([panel.panel_ready, panel.data_ready], timeout=2000, raising=True):
            panel.begin_async_init()

        assert panel.canvas is not None
        assert panel.results_widget is not None

    def test_start_analysis_builds_wizard_without_crashing(self, qtbot):
        """Regression test: the step classes emit directly on the wizard
        instance (self._panel.image_changed.emit(...), etc.), so the wizard
        built here must actually carry those signals — a plain
        karcytics_sdk.plugin.wizard.WizardPanel does not declare them, only
        WesternBlotPanel's own _WesternBlotWizardPanel subclass does.
        """
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)

        with qtbot.waitSignals([panel.panel_ready, panel.data_ready], timeout=2000, raising=True):
            panel.begin_async_init()

        panel._on_start_analysis(include_ponceau=False)

    def test_start_analysis_with_ponceau_builds_wizard_without_crashing(self, qtbot):
        """Same as above, but with the optional Ponceau stage included — its
        three extra steps (PonceauLoadStep, PonceauLanesStep, PonceauBandsStep)
        get build_page()'d eagerly too, at wizard construction time.
        """
        panel = WesternBlotPanel()
        qtbot.addWidget(panel)

        with qtbot.waitSignals([panel.panel_ready, panel.data_ready], timeout=2000, raising=True):
            panel.begin_async_init()

        panel._on_start_analysis(include_ponceau=True)
