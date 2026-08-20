"""Ponceau Stain — Step 1: Load & Preprocess."""

from karcytics_sdk.plugin import WizardPanel
from karcytics_sdk.plugin.theme_fallback import Colors

from .base_load_step import BaseLoadStep


class PonceauLoadStep(BaseLoadStep):
    """Load and preprocess the Ponceau S stain image."""

    label = "Pon. Load"
    _default_contrast = 2.5
    _min_contrast = 0.5
    _banner_text = (
        "📌  Load the Ponceau S stain image of the same membrane.\n"
        "This is used to measure total protein per lane for loading normalisation.\n"
        "The WB image is loaded in the next stage."
    )
    _banner_style = (
        f"background: {Colors.BG_DARK}; color: {Colors.FG_SECONDARY};"
        f" border: 1px solid {Colors.BORDER}; border-radius: 6px;"
        f" padding: 10px; font-size: 11px;"
    )
    _file_group_title = "Ponceau S Image"
    _open_btn_text = "📁  Open Ponceau Image File..."
    _default_brightness = -0.5

    def get_analyzer(self, panel: WizardPanel):
        return panel.ponceau_analyzer

    def get_smart_contrast(self, image) -> tuple[float, float]:
        import numpy as np

        flat = image.ravel()
        p2 = float(np.percentile(flat, 2))
        p98 = float(np.percentile(flat, 98))
        span = p98 - p2
        if span < 1e-6:
            return 2.5, -0.7
        alpha = round(1.0 / span, 3)
        beta = round(-p2 / span, 3)
        alpha = float(np.clip(max(alpha, 2.0), 0.5, 10.0))
        beta = float(np.clip(beta, -2.0, 2.0))
        return alpha, beta

    def _post_load_hook(self) -> None:
        self._on_auto_contrast()

    def on_enter(self) -> None:
        pass

    def on_next(self, panel: WizardPanel) -> bool:
        if panel.ponceau_analyzer.state.original_image is None:
            panel.status_message.emit("Please load a Ponceau image first.")
            return False
        self._preprocess()
        return True

    def _get_status_prefix(self) -> str:
        return "Ponceau: "

    def _get_open_dialog_title(self) -> str:
        return "Open Ponceau S Image"

    def _get_invert_checkbox_text(self) -> str:
        return "Auto-invert (pink bands on white background)"

    def _get_invert_tooltip(self) -> str:
        return "Ponceau S bands are pink on white — after grayscale conversion\nthey appear as dark bands on a light background.\nAuto-invert detects this and flips appropriately."

    def _get_rotation_tooltip(self) -> str:
        return "Rotate image. Positive = counter-clockwise."

    def _get_contrast_tooltip(self) -> str:
        return "Contrast multiplier. Ponceau bands are faint — start high (2–4×).\noutput = α × pixel + β"

    def _get_brightness_tooltip(self) -> str:
        return "Brightness offset. Negative darkens background — helps separate\nfaint Ponceau bands from the white membrane background."

    def _get_auto_hint(self) -> str:
        return "Auto-detect sets rotation and contrast for this image. Because Ponceau bands are faint, you may want to increase contrast further manually after auto-detect."

    def _get_auto_rotation_tooltip(self) -> str:
        return "Auto-detect optimal rotation angle."

    def _get_auto_contrast_tooltip(self) -> str:
        return "Auto-compute contrast/brightness for Ponceau (faint pink bands)."

    def _get_manual_crop_status_msg(self) -> str:
        return "Crop mode: draw a rectangle on the full Ponceau image."
