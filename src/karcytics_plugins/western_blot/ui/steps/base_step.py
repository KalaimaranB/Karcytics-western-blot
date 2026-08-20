"""Base class for all Wizard steps to eliminate boilerplate."""

from karcytics_sdk.plugin import HeaderLabel, SubtitleLabel
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class BaseStepWidget(QWidget):
    """Provides standard margins, typography, and a content container."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)

        # 1. Standardized Outer Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # 2. Standardized Header
        self.lbl_title = HeaderLabel(title)
        self.main_layout.addWidget(self.lbl_title)

        if subtitle:
            self.lbl_subtitle = SubtitleLabel(subtitle)
            self.lbl_subtitle.setWordWrap(True)
            self.main_layout.addWidget(self.lbl_subtitle)

        # 3. Dynamic Content Container (Children put their specific UI here!)
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(10)
        self.main_layout.addLayout(self.content_layout)

        # 4. Push everything to the top
        self.main_layout.addStretch()

    def add_content_widget(self, widget: QWidget):
        """Helper to add child widgets directly into the content zone."""
        self.content_layout.addWidget(widget)
