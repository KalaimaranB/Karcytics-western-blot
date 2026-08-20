"""Base class for Load & Preprocess Wizard steps.

Consolidates the massive duplication between WB Load and Ponceau Load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from karcytics_sdk.plugin import PrimaryButton, WizardPanel, WizardStep, get_image_path, show_error
from karcytics_sdk.plugin.theme_fallback import Colors
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class BaseLoadStep(WizardStep):
    """Abstract base class for load steps (Ponceau and WB)."""

    label = "Base Load"

    # Defaults intended to be overridden
    _default_contrast = 1.5
    _min_contrast = 0.5
    _banner_text = ""
    _banner_style = ""
    _file_group_title = "Image File"
    _open_btn_text = "📁  Open Image File..."
    _default_brightness = -0.7

    def get_analyzer(self, panel: WizardPanel):
        """Must be implemented by subclasses to return the correct analyzer."""
        raise NotImplementedError

    def get_smart_contrast(self, image) -> tuple[float, float]:
        """Must be implemented by subclasses to compute auto contrast."""
        raise NotImplementedError

    def _get_status_prefix(self) -> str:
        """Returns a prefix for status messages."""
        return ""

    def build_page(self, panel: WizardPanel) -> QWidget:
        self._panel = panel
        self._canvas = None
        self._pending_crop_rect: tuple[int, int, int, int] | None = None

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        if self._banner_text:
            banner = QLabel(self._banner_text)
            banner.setWordWrap(True)
            if self._banner_style:
                banner.setStyleSheet(self._banner_style)
            layout.addWidget(banner)

        # File picker
        file_group = QGroupBox(self._file_group_title)
        file_layout = QVBoxLayout(file_group)
        self.btn_open = PrimaryButton(self._open_btn_text)
        self.btn_open.clicked.connect(self._open_file)
        file_layout.addWidget(self.btn_open)
        self.lbl_filename = QLabel("No file loaded")
        self.lbl_filename.setObjectName("subtitle")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setMinimumHeight(18)
        file_layout.addWidget(self.lbl_filename)
        layout.addWidget(file_group)

        # Live adjustments
        live_group = QGroupBox("Live Adjustments  —  preview updates as you type")
        live_layout = QVBoxLayout(live_group)
        live_layout.setSpacing(8)

        self.chk_invert = QCheckBox(self._get_invert_checkbox_text())
        self.chk_invert.setChecked(True)
        self.chk_invert.setToolTip(self._get_invert_tooltip())
        self.chk_invert.toggled.connect(self._on_preprocess_changed)
        live_layout.addWidget(self.chk_invert)

        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-180, 180)
        self.spin_rotation.setValue(0)
        self.spin_rotation.setSuffix("°")
        self.spin_rotation.setSingleStep(0.5)
        self.spin_rotation.setToolTip(self._get_rotation_tooltip())
        self.spin_rotation.valueChanged.connect(self._on_rotation_changed)
        live_layout.addLayout(self._row("Rotation:", self.spin_rotation))

        rot_btn_row = QHBoxLayout()
        rot_btn_row.setSpacing(4)
        for lbl, delta in [("-90°", -90), ("-45°", -45), ("+45°", 45), ("+90°", 90)]:
            btn = QPushButton(lbl)
            btn.setMinimumHeight(28)
            btn.setToolTip(f"Add {delta}° to current rotation")
            btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY};"
                f" border: 1px solid {Colors.BORDER}; border-radius: 5px;"
                f" padding: 3px 6px; font-size: 11px; }}"
                f"QPushButton:hover {{ background: {Colors.BG_LIGHT}; }}"
            )
            btn.clicked.connect(lambda _, d=delta: self._rotate_by(d))
            rot_btn_row.addWidget(btn)
        live_layout.addLayout(rot_btn_row)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(self._min_contrast, 10.0)
        self.spin_contrast.setValue(self._default_contrast)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setToolTip(self._get_contrast_tooltip())
        self.spin_contrast.valueChanged.connect(self._on_contrast_manually_changed)
        live_layout.addLayout(self._row("Contrast (α):", self.spin_contrast))

        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setRange(-2.0, 2.0)
        self.spin_brightness.setValue(self._default_brightness)
        self.spin_brightness.setSingleStep(0.05)
        self.spin_brightness.setDecimals(3)
        self.spin_brightness.setToolTip(self._get_brightness_tooltip())
        self.spin_brightness.valueChanged.connect(self._on_preprocess_changed)
        live_layout.addLayout(self._row("Brightness (β):", self.spin_brightness))

        self.btn_reset = QPushButton("↩  Reset to Defaults")
        self.btn_reset.setToolTip("Reset rotation, contrast and brightness to defaults.")
        self.btn_reset.clicked.connect(self._on_reset_preprocess)
        live_layout.addWidget(self.btn_reset)
        layout.addWidget(live_group)

        # Smart auto-detect
        auto_group = QGroupBox("Smart Auto-detect")
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setSpacing(8)

        hint = QLabel(self._get_auto_hint())
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        hint.setMinimumHeight(32)
        auto_layout.addWidget(hint)

        auto_btn_row = QHBoxLayout()
        auto_btn_row.setSpacing(6)

        self.btn_auto_rotation = QPushButton("🔄  Auto Rotation")
        self.btn_auto_rotation.setMinimumHeight(36)
        self.btn_auto_rotation.setToolTip(self._get_auto_rotation_tooltip())
        self.btn_auto_rotation.clicked.connect(self._on_auto_rotation)
        auto_btn_row.addWidget(self.btn_auto_rotation)

        self.btn_auto_contrast = QPushButton("🎨  Auto Contrast")
        self.btn_auto_contrast.setMinimumHeight(36)
        self.btn_auto_contrast.setToolTip(self._get_auto_contrast_tooltip())
        self.btn_auto_contrast.clicked.connect(self._on_auto_contrast)
        auto_btn_row.addWidget(self.btn_auto_contrast)

        auto_layout.addLayout(auto_btn_row)

        self.btn_auto_crop = QPushButton("✂️  Auto-crop to Band Region")
        self.btn_auto_crop.setMinimumHeight(36)
        self.btn_auto_crop.setToolTip(self._get_auto_crop_tooltip())
        self.btn_auto_crop.clicked.connect(self._on_auto_crop_bands)
        auto_layout.addWidget(self.btn_auto_crop)

        confirm_row = QHBoxLayout()
        self.btn_confirm_crop = QPushButton("✅  Confirm Crop")
        self.btn_confirm_crop.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST};"
            f" border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; }}"
            f"QPushButton:hover {{ background-color: {Colors.ACCENT_PRIMARY_HOVER}; }}"
            f"QPushButton:pressed {{ background-color: {Colors.ACCENT_PRIMARY_PRESSED}; }}"
        )
        self.btn_confirm_crop.setMinimumHeight(34)
        self.btn_confirm_crop.setVisible(False)
        self.btn_confirm_crop.clicked.connect(self._on_confirm_crop)
        confirm_row.addWidget(self.btn_confirm_crop)

        self.btn_cancel_crop = QPushButton("✖  Cancel")
        self.btn_cancel_crop.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY};"
            f" border: 1px solid {Colors.BORDER}; border-radius: 6px; padding: 7px 14px; }}"
            f"QPushButton:hover {{ background-color: {Colors.BG_LIGHT}; }}"
        )
        self.btn_cancel_crop.setMinimumHeight(34)
        self.btn_cancel_crop.setVisible(False)
        self.btn_cancel_crop.clicked.connect(self._on_cancel_crop)
        confirm_row.addWidget(self.btn_cancel_crop)
        auto_layout.addLayout(confirm_row)

        self.lbl_auto_result = QLabel("")
        self.lbl_auto_result.setObjectName("subtitle")
        self.lbl_auto_result.setWordWrap(True)
        self.lbl_auto_result.setMinimumHeight(18)
        auto_layout.addWidget(self.lbl_auto_result)
        layout.addWidget(auto_group)

        # Manual crop
        crop_group = QGroupBox("Manual Crop")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.setSpacing(8)

        crop_hint = QLabel(self._get_manual_crop_hint())
        crop_hint.setWordWrap(True)
        crop_hint.setObjectName("subtitle")
        crop_hint.setMinimumHeight(18)
        crop_layout.addWidget(crop_hint)

        self.btn_manual_crop = QPushButton("✂️  Start Manual Crop")
        self.btn_manual_crop.setCheckable(True)
        self.btn_manual_crop.setMinimumHeight(34)
        self.btn_manual_crop.setToolTip(self._get_start_manual_crop_tooltip())
        self.btn_manual_crop.toggled.connect(self._on_manual_crop_toggled)
        crop_layout.addWidget(self.btn_manual_crop)

        self.btn_clear_crop = QPushButton("🗑  Clear Crop")
        self.btn_clear_crop.setMinimumHeight(34)
        self.btn_clear_crop.setToolTip(self._get_clear_crop_tooltip())
        self.btn_clear_crop.clicked.connect(self._on_clear_crop)
        crop_layout.addWidget(self.btn_clear_crop)
        layout.addWidget(crop_group)

        layout.addStretch()
        return self._scroll(page)

    def set_canvas(self, canvas) -> None:
        self._canvas = canvas

    def on_enter(self) -> None:
        """Architectural Pillar 1 & 2:
        Strict Unidirectional State Hydration.
        When this step becomes active, it MUST explicitly read the backend state
        and force the UI + Canvas to match it.
        """
        if not hasattr(self, "_panel") or not self._panel:
            return

        analyzer = self.get_analyzer(self._panel)
        if not analyzer or not analyzer.state:
            return

        # 1. Sync the UI Spinners and Checkboxes to the math backend
        self.spin_rotation.blockSignals(True)
        self.spin_contrast.blockSignals(True)
        self.spin_brightness.blockSignals(True)
        self.chk_invert.blockSignals(True)

        # Handle the "auto" vs boolean inversion logic
        is_inverted = getattr(analyzer.state, "is_inverted", False)
        self.chk_invert.setChecked(is_inverted if isinstance(is_inverted, bool) else True)

        self.spin_rotation.setValue(getattr(analyzer.state, "rotation_angle", 0.0))
        self.spin_contrast.setValue(
            getattr(analyzer.state, "contrast_alpha", self._default_contrast)
        )
        self.spin_brightness.setValue(
            getattr(analyzer.state, "contrast_beta", self._default_brightness)
        )

        self.spin_rotation.blockSignals(False)
        self.spin_contrast.blockSignals(False)
        self.spin_brightness.blockSignals(False)
        self.chk_invert.blockSignals(False)

        # 2. Sync the File Status Label
        img_path = getattr(analyzer.state, "image_path", None)
        if img_path:
            import os

            filename = os.path.basename(str(img_path))
            self.lbl_filename.setText(f"✅  {filename}")
        else:
            self.lbl_filename.setText("No file loaded")

        # 3. Explicit Canvas Context Switching (Pillar 2)
        # We must broadcast the active analyzer's data to the shared canvas
        if analyzer.state.processed_image is not None:
            self._panel.image_changed.emit(analyzer.state.processed_image)
        elif analyzer.state.original_image is not None:
            # Fallback if preprocessing hasn't happened yet
            self._panel.image_changed.emit(analyzer.state.original_image)

        # Since this is the Load step, we don't draw lanes/bands yet.
        # But we MUST clear any leftover lanes/bands from the previous step!
        self._panel.lanes_detected.emit([])
        self._panel.bands_detected.emit([], [])

        # 4. Final safety check: if we have an image but NO processed image, force a preprocess
        if img_path and analyzer.state.processed_image is None:
            self._preprocess()

    def _open_file(self) -> None:
        # karcytics_sdk.plugin.dialogs.get_image_path() has no `filters`
        # parameter — it hardcodes an image-extension filter (tif/png/jpg,
        # no bmp). Passing filters= here raised TypeError on every call.
        path = get_image_path(
            self._panel,
            self._get_open_dialog_title(),
        )
        if not path:
            return

        final_path = Path(path)

        try:
            analyzer = self.get_analyzer(self._panel)
            analyzer.load_image(str(final_path))
            self.lbl_filename.setText(f"✅  {final_path.name}")
            prefix = self._get_status_prefix()
            self._panel.status_message.emit(f"{prefix}Loaded {final_path.name}")

            self._post_load_hook()

        except Exception as e:
            self.lbl_filename.setText(f"❌  Error: {e}")
            prefix = self._get_status_prefix()
            self._panel.status_message.emit(f"Error loading {prefix}image: {e}")
            show_error(self._panel, f"Failed to load {prefix}image", str(e))
            logger.exception(f"Error loading {prefix}image")

    def _post_load_hook(self) -> None:
        """Called after an image successfully loads."""
        pass

    def _preprocess(self) -> None:
        analyzer = self.get_analyzer(self._panel)
        if analyzer.state.original_image is None:
            return

        # ── THE FIX: Force the UI to write its live settings into backend memory ──
        analyzer.state.is_inverted = self.chk_invert.isChecked()
        analyzer.state.rotation_angle = self.spin_rotation.value()
        analyzer.state.contrast_alpha = self.spin_contrast.value()
        analyzer.state.contrast_beta = self.spin_brightness.value()
        # ──────────────────────────────────────────────────────────────────────────

        try:
            processed = analyzer.preprocess(
                invert_lut="auto" if self.chk_invert.isChecked() else False,
                rotation_angle=self.spin_rotation.value(),
                contrast_alpha=self.spin_contrast.value(),
                contrast_beta=self.spin_brightness.value(),
                manual_crop_rect=analyzer.state.manual_crop_rect,
            )
            self._panel.image_changed.emit(processed)

            parts = []
            if analyzer.state.is_inverted:
                parts.append("inverted")
            rot = self.spin_rotation.value()
            if abs(rot) > 0.01:
                parts.append(f"rotated {rot:.1f}°")
            alpha = self.spin_contrast.value()
            beta = self.spin_brightness.value()
            if abs(alpha - 1.0) > 0.01 or abs(beta) > 0.001:
                parts.append(f"contrast ×{alpha:.2f}{beta:+.3f}")
            suffix = f" ({', '.join(parts)})" if parts else ""
            prefix = self._get_status_prefix()
            self._panel.status_message.emit(f"{prefix}Preprocessed{suffix}")
            self._panel.state_changed.emit()
        except Exception as e:
            self._panel.status_message.emit(f"Preprocessing error: {e}")
            logger.exception("Preprocessing error")

    def _on_preprocess_changed(self, *_) -> None:
        if self.get_analyzer(self._panel).state.original_image is None:
            return
        self._preprocess()

    def _on_rotation_changed(self, *_) -> None:
        self._on_preprocess_changed()

    def _on_contrast_manually_changed(self, *_) -> None:
        self._on_preprocess_changed()

    def _on_reset_preprocess(self) -> None:
        for spin in (self.spin_rotation, self.spin_contrast, self.spin_brightness):
            spin.blockSignals(True)
        self.spin_rotation.setValue(0.0)
        self.spin_contrast.setValue(self._default_contrast)
        self.spin_brightness.setValue(self._default_brightness)
        for spin in (self.spin_rotation, self.spin_contrast, self.spin_brightness):
            spin.blockSignals(False)
        self.lbl_auto_result.setText("")
        self._on_preprocess_changed()

    def _rotate_by(self, delta: float) -> None:
        current = self.spin_rotation.value()
        new_val = (current + delta + 180) % 360 - 180
        self.spin_rotation.setValue(round(new_val, 1))

    def _on_auto_rotation(self) -> None:
        analyzer = self.get_analyzer(self._panel)
        if analyzer.state.original_image is None:
            self._panel.status_message.emit("Load an image first.")
            return
        try:
            from karcytics_sdk.contrib.image_utils import auto_detect_rotation

            self.lbl_auto_result.setText("⏳  Detecting rotation…")
            self.btn_auto_rotation.setEnabled(False)
            self.btn_auto_rotation.repaint()

            image = analyzer.state.original_image
            alpha = self.spin_contrast.value()
            beta = self.spin_brightness.value()
            stretched = np.clip(image * alpha + beta, 0.0, 1.0)
            angle = auto_detect_rotation(stretched)

            self.spin_rotation.blockSignals(True)
            self.spin_rotation.setValue(round(angle, 2))
            self.spin_rotation.blockSignals(False)

            msg = f"✅  Rotation: {angle:+.2f}°"
            self.lbl_auto_result.setText(msg)
            prefix = self._get_status_prefix()
            self._panel.status_message.emit(f"{prefix}auto-rotation: {msg}")
            self._preprocess()
        except Exception as e:
            self.lbl_auto_result.setText(f"❌  Rotation detection failed: {e}")
            logger.exception("Auto-rotation error")
        finally:
            self.btn_auto_rotation.setEnabled(True)

    def _on_auto_contrast(self) -> None:
        analyzer = self.get_analyzer(self._panel)
        if analyzer.state.original_image is None:
            self._panel.status_message.emit("Load an image first.")
            return
        try:
            self.lbl_auto_result.setText("⏳  Computing contrast…")
            self.btn_auto_contrast.setEnabled(False)
            self.btn_auto_contrast.repaint()

            image = analyzer.state.original_image
            alpha, beta = self.get_smart_contrast(image)

            self.spin_contrast.blockSignals(True)
            self.spin_brightness.blockSignals(True)
            self.spin_contrast.setValue(round(alpha, 2))
            self.spin_brightness.setValue(round(beta, 3))
            self.spin_contrast.blockSignals(False)
            self.spin_brightness.blockSignals(False)

            msg = f"✅  Contrast: ×{alpha:.2f}, β={beta:+.3f}"
            self.lbl_auto_result.setText(msg)
            prefix = self._get_status_prefix()
            self._panel.status_message.emit(f"{prefix}auto-contrast complete — {msg}")
            self._preprocess()
        except Exception as e:
            self.lbl_auto_result.setText(f"❌  Contrast detection failed: {e}")
            logger.exception("Auto-contrast error")
        finally:
            self.btn_auto_contrast.setEnabled(True)

    def _on_auto_crop_bands(self) -> None:
        analyzer = self.get_analyzer(self._panel)
        if analyzer.state.processed_image is None:
            self._panel.status_message.emit("Load and preprocess an image first.")
            return
        try:
            from karcytics_sdk.contrib.image_utils import calculate_band_crop_region

            self.btn_auto_crop.setEnabled(False)
            self.btn_auto_crop.repaint()

            image = analyzer.state.base_image
            if image is None:
                image = analyzer.state.processed_image
            region = calculate_band_crop_region(
                image,
                dark_threshold=0.85,
                min_band_width_frac=0.01,
                min_band_height_frac=0.01,
                vertical_padding_frac=0.15,
                horizontal_padding_frac=0.10,
                smoothing_window=9,
            )
            if region is None or (hasattr(region, "__len__") and len(region) == 0):
                self.lbl_auto_result.setText("⚠️  No band region detected.")
                self._panel.status_message.emit("Auto-crop failed: no band region found.")
                return

            r_min, r_max, c_min, c_max = (int(v) for v in region)
            if r_min >= r_max or c_min >= c_max:
                self.lbl_auto_result.setText("⚠️  No valid band region found.")
                self._panel.status_message.emit("Auto-crop failed: invalid region.")
                return

            self._pending_crop_rect = (r_min, r_max, c_min, c_max)
            if self._canvas is not None:
                self._canvas.show_crop_preview(QRectF(c_min, r_min, c_max - c_min, r_max - r_min))
            self.btn_confirm_crop.setVisible(True)
            self.btn_cancel_crop.setVisible(True)
            crop_w, crop_h = c_max - c_min, r_max - r_min
            self.lbl_auto_result.setText(f"📐  Preview: {crop_w}×{crop_h} px. Confirm to apply.")
            self._panel.status_message.emit(
                f"Crop preview — {crop_w}×{crop_h} px. Confirm or cancel."
            )
        except Exception as e:
            self.lbl_auto_result.setText(f"❌  Error: {e}")
            self._panel.status_message.emit(f"Auto-crop error: {e}")
            logger.exception("Auto-crop error")
        finally:
            self.btn_auto_crop.setEnabled(True)

    def _on_confirm_crop(self) -> None:
        if self._pending_crop_rect is None:
            return
        try:
            bounds = None
            if self._canvas is not None and hasattr(
                self._canvas, "get_current_crop_preview_bounds"
            ):
                bounds = self._canvas.get_current_crop_preview_bounds()
            r_min, r_max, c_min, c_max = bounds if bounds else self._pending_crop_rect

            analyzer = self.get_analyzer(self._panel)
            base = analyzer.state.base_image
            image = base if base is not None else analyzer.state.processed_image
            h, w = image.shape[:2]
            r_min = max(0, min(r_min, h - 1))
            r_max = max(r_min + 1, min(r_max, h))
            c_min = max(0, min(c_min, w - 1))
            c_max = max(c_min + 1, min(c_max, w))

            analyzer.state.manual_crop_rect = (c_min, r_min, c_max - c_min, r_max - r_min)
            self._preprocess()
            self.lbl_auto_result.setText(f"✅  Cropped to {c_max - c_min}×{r_max - r_min} px.")
            self._panel.status_message.emit("Band region crop applied.")
        except Exception as e:
            self._panel.status_message.emit(f"Crop error: {e}")
            logger.exception("Confirm crop error")
        finally:
            self._pending_crop_rect = None
            self.btn_confirm_crop.setVisible(False)
            self.btn_cancel_crop.setVisible(False)
            if self._canvas is not None:
                self._canvas.clear_crop_preview()

    def _on_cancel_crop(self) -> None:
        self._pending_crop_rect = None
        self.btn_confirm_crop.setVisible(False)
        self.btn_cancel_crop.setVisible(False)
        if self._canvas is not None:
            self._canvas.clear_crop_preview()
        self.lbl_auto_result.setText("Crop cancelled.")
        self._panel.status_message.emit("Crop cancelled.")

    def _on_manual_crop_toggled(self, checked: bool) -> None:
        self._panel.crop_mode_toggled.emit(checked)
        analyzer = self.get_analyzer(self._panel)
        if checked:
            base = analyzer.state.base_image
            if base is not None:
                self._panel.image_changed.emit(base)
            crop = analyzer.state.manual_crop_rect
            if crop is not None and self._canvas is not None:
                from PyQt6.QtCore import QRectF

                x, y, w, h = crop
                self._canvas.show_crop_preview(QRectF(x, y, w, h))
            self._panel.status_message.emit(self._get_manual_crop_status_msg())
        else:
            if self._canvas is not None:
                self._canvas.clear_crop_preview()
            processed = analyzer.state.processed_image
            if processed is not None:
                self._panel.image_changed.emit(processed)
            self._panel.status_message.emit("Manual crop cancelled.")

    def on_crop_requested(self, rect, panel: WizardPanel) -> None:
        x, y = int(round(rect.x())), int(round(rect.y()))
        w, h = int(round(rect.width())), int(round(rect.height()))
        self.get_analyzer(panel).state.manual_crop_rect = (x, y, w, h)
        self.btn_manual_crop.setChecked(False)
        self._preprocess()

    def _on_clear_crop(self) -> None:
        self.get_analyzer(self._panel).state.manual_crop_rect = None
        self.btn_manual_crop.setChecked(False)
        self._preprocess()
        self._panel.status_message.emit("Crop cleared — showing full image.")

    # Override hooks returning strings
    def _get_invert_checkbox_text(self) -> str:
        return "Auto-invert"

    def _get_invert_tooltip(self) -> str:
        return "Auto-invert image"

    def _get_rotation_tooltip(self) -> str:
        return "Rotate image"

    def _get_contrast_tooltip(self) -> str:
        return "Contrast multiplier"

    def _get_brightness_tooltip(self) -> str:
        return "Brightness offset"

    def _get_auto_hint(self) -> str:
        return "Click the buttons below to automatically compute optimal values."

    def _get_auto_rotation_tooltip(self) -> str:
        return "Auto rotation."

    def _get_auto_contrast_tooltip(self) -> str:
        return "Auto contrast."

    def _get_auto_crop_tooltip(self) -> str:
        return "Auto crop to band region."

    def _get_manual_crop_hint(self) -> str:
        return "Draw a rectangle directly on the image to crop it."

    def _get_start_manual_crop_tooltip(self) -> str:
        return "Click to enter crop mode."

    def _get_clear_crop_tooltip(self) -> str:
        return "Remove crop"

    def _get_open_dialog_title(self) -> str:
        return "Open Image"

    def _get_manual_crop_status_msg(self) -> str:
        return "Crop mode: draw a rectangle."
