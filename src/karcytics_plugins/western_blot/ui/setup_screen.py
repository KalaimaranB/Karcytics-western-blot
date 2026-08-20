"""Western Blot pipeline setup screen.

Shown once when the user enters Western Blot from the home screen,
before the wizard begins.  The user chooses which optional stages to
include (currently: Ponceau stain normalization), then clicks
"Start Analysis" to build the correct step list and launch the wizard.

Adding a new option in future means adding one checkbox here and one
``if checkbox.isChecked(): steps.insert(…)`` in ``_on_start``.
"""

from __future__ import annotations

from karcytics_sdk.plugin.theme_fallback import Colors, Fonts
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _OptionCard(QFrame):
    """A single pipeline option presented as a card with checkbox."""

    def __init__(
        self,
        title: str,
        description: str,
        *,
        checked: bool = False,
        recommended: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("optionCard")
        self._apply_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Explicit style so the indicator is visible on the dark background
        self.checkbox.setStyleSheet(
            f"QCheckBox::indicator {{ width: 18px; height: 18px;"
            f" border: 2px solid {Colors.BORDER_FOCUS}; border-radius: 4px;"
            f" background: {Colors.BG_MEDIUM}; }}"
            f"QCheckBox::indicator:checked {{ background: {Colors.ACCENT_PRIMARY};"
            f" border-color: {Colors.ACCENT_PRIMARY};"
            f" image: url(none); }}"
            f"QCheckBox::indicator:unchecked:hover {{ border-color: {Colors.FG_SECONDARY}; }}"
        )
        layout.addWidget(self.checkbox, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        title_row = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: {Fonts.SIZE_NORMAL}px; font-weight: 700;"
            f" color: {Colors.FG_PRIMARY}; background: transparent;"
        )
        title_row.addWidget(title_lbl)

        if recommended:
            rec_lbl = QLabel("Recommended")
            rec_lbl.setStyleSheet(
                f"background: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST};"
                f" border-radius: 4px; padding: 1px 6px;"
                f" font-size: 10px; font-weight: 700;"
            )
            title_row.addWidget(rec_lbl)
        title_row.addStretch()
        text_col.addLayout(title_row)

        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(
            f"font-size: {Fonts.SIZE_SMALL}px; color: {Colors.FG_SECONDARY};"
            f" background: transparent;"
        )
        text_col.addWidget(desc_lbl)
        layout.addLayout(text_col)

        # Clicking anywhere on the card toggles the checkbox
        self.mousePressEvent = lambda _e: self.checkbox.toggle()  # type: ignore[method-assign,assignment]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"QFrame#optionCard {{ background: {Colors.BG_DARK};"
            f" border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
            f"QFrame#optionCard:hover {{ border-color: {Colors.FG_SECONDARY}; }}"
        )

    @property
    def is_checked(self) -> bool:
        return self.checkbox.isChecked()


class SetupScreen(QWidget):
    """Pre-wizard configuration screen for Western Blot analysis.

    Signals:
        analysis_requested(include_ponceau): Emitted when the user clicks
            "Start Analysis".  Carries a bool indicating whether the
            Ponceau stain stage should be included.
    """

    analysis_requested = pyqtSignal(bool)  # include_ponceau

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background: {Colors.BG_DARK}; border-bottom: 1px solid {Colors.BORDER};"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 14)

        title = QLabel("🔬  Western Blot Analysis")
        title.setObjectName("stepTitle")
        header_layout.addWidget(title)

        subtitle = QLabel("Choose which pipeline stages to include for this run.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"font-size: {Fonts.SIZE_SMALL}px; color: {Colors.FG_SECONDARY};"
            f" background: transparent;"
        )
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        # ── Content ───────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background: {Colors.BG_DARKEST};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 16, 20, 16)
        content_layout.setSpacing(12)

        # Always-on stage label
        always_lbl = QLabel("Always included")
        always_lbl.setStyleSheet(
            f"font-size: {Fonts.SIZE_SMALL}px; font-weight: 600;"
            f" color: {Colors.FG_SECONDARY}; text-transform: uppercase;"
            f" letter-spacing: 1px;"
        )
        content_layout.addWidget(always_lbl)

        wb_card = _OptionCard(
            title="Western Blot Densitometry",
            description=("Load image → detect lanes → detect bands → compute relative densities."),
            checked=True,
        )
        wb_card.checkbox.setEnabled(False)  # cannot be unchecked
        wb_card.setStyleSheet(
            f"QFrame#optionCard {{ background: {Colors.BG_DARK};"
            f" border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        content_layout.addWidget(wb_card)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {Colors.BORDER};")
        content_layout.addWidget(sep)

        # Optional stages label
        optional_lbl = QLabel("Optional stages")
        optional_lbl.setStyleSheet(
            f"font-size: {Fonts.SIZE_SMALL}px; font-weight: 600;"
            f" color: {Colors.FG_SECONDARY}; text-transform: uppercase;"
            f" letter-spacing: 1px;"
        )
        content_layout.addWidget(optional_lbl)

        self._ponceau_card = _OptionCard(
            title="Ponceau Stain Normalization",
            description=(
                "Load a Ponceau S stain image of the same membrane, detect its bands, "
                "and use the lane intensities to normalize Western Blot densities for "
                "unequal loading.  Produces more reliable quantification — "
                "skip only if no Ponceau image is available."
            ),
            checked=True,  # on by default — it IS recommended
            recommended=True,
        )
        content_layout.addWidget(self._ponceau_card)

        content_layout.addStretch()

        # ── Start button ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_start = QPushButton("Start Analysis →")
        self._btn_start.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_PRIMARY};"
            f" color: {Colors.BG_DARKEST}; border: none; border-radius: 6px;"
            f" padding: 10px 28px; font-size: {Fonts.SIZE_NORMAL}px; font-weight: 700; }}"
            f"QPushButton:hover {{ background-color: {Colors.ACCENT_PRIMARY_HOVER}; }}"
            f"QPushButton:pressed {{ background-color: {Colors.ACCENT_PRIMARY_PRESSED}; }}"
        )
        self._btn_start.setMinimumHeight(42)
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)
        content_layout.addLayout(btn_row)

        root.addWidget(content, stretch=1)

    def _on_start(self) -> None:
        self.analysis_requested.emit(self._ponceau_card.is_checked)
