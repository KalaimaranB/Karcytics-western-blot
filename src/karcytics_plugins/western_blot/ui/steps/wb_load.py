"""Western Blot — Step 1: Load & Preprocess."""

from karcytics_sdk.plugin import WizardPanel

from .base_load_step import BaseLoadStep


class WBLoadStep(BaseLoadStep):
    """Load an image and apply preprocessing (inversion, rotation, crop)."""

    label = "Load"

    def get_analyzer(self, panel: WizardPanel):
        return panel.analyzer

    def get_smart_contrast(self, image) -> tuple[float, float]:
        import numpy as np

        flat = image.ravel()
        p2 = float(np.percentile(flat, 2))
        p98 = float(np.percentile(flat, 98))
        span = p98 - p2
        if span < 1e-6:
            return 1.5, -0.7
        alpha = round(1.0 / span, 3)
        beta = round(-p2 / span, 3)
        alpha = float(np.clip(alpha, 0.5, 8.0))
        beta = float(np.clip(beta, -2.0, 2.0))
        return alpha, beta

    def _post_load_hook(self) -> None:
        self._preprocess()

    def on_enter(self) -> None:
        if self._panel.analyzer.state.processed_image is not None:
            self._panel.image_changed.emit(self._panel.analyzer.state.processed_image)
            return

        import numpy as np

        blank_image = np.zeros((800, 800), dtype=np.float64)
        self._panel.image_changed.emit(blank_image)
        self._panel.lanes_detected.emit([])
        self._panel.bands_detected.emit([], [])
        self._panel.status_message.emit("Please load your Western Blot image.")

    def on_next(self, panel: WizardPanel) -> bool:
        analyzer = panel.analyzer
        if analyzer.state.original_image is None:
            panel.status_message.emit("Please load an image first.")
            return False
        self._preprocess()
        from .wb_lanes import WBLanesStep

        for step in panel._steps:
            if isinstance(step, WBLanesStep) and step._auto_lanes_checked():
                step.run_detection(panel)
                break
        return True

    def _get_status_prefix(self) -> str:
        return "Western Blot: "

    def _get_open_dialog_title(self) -> str:
        return "Open Western Blot Image"

    def _get_invert_checkbox_text(self) -> str:
        return "Auto-invert (dark bands on white background)"

    def _get_invert_tooltip(self) -> str:
        return "When checked, the image is automatically inverted if needed so that\ndark bands on a white background become detectable peaks.\nUncheck if your image is already the right way around."

    def _get_rotation_tooltip(self) -> str:
        return "Rotates the image in real-time. Positive = counter-clockwise."

    def _get_contrast_tooltip(self) -> str:
        return "Contrast multiplier. output = α × pixel + β\n>1.0 = more contrast, <1.0 = less."

    def _get_brightness_tooltip(self) -> str:
        return "Brightness offset. output = α × pixel + β\nNegative = shift darker (useful for high-background images)."

    def _get_auto_rotation_tooltip(self) -> str:
        return "Automatically detect and apply the optimal rotation angle."

    def _get_auto_contrast_tooltip(self) -> str:
        return "Automatically compute optimal contrast (α) and brightness (β)\nusing percentile-based stretching for blot images."

    def _get_auto_crop_tooltip(self) -> str:
        return "Detects where the bands are and shows a preview outline.\nClick 'Confirm Crop' to apply, or 'Cancel' to discard."

    def _get_start_manual_crop_tooltip(self) -> str:
        return "Click to enter crop mode, then drag on the image to draw a rectangle."

    def _get_clear_crop_tooltip(self) -> str:
        return "Remove the current crop and restore the full image."

    def _get_manual_crop_status_msg(self) -> str:
        return "Crop mode: drag the handles to adjust, or draw a new rectangle. Click Confirm to apply."
