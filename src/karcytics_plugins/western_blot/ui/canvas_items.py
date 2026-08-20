"""Interactive graphical items for the image canvas overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING

from karcytics_sdk.plugin.theme_fallback import Colors
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsRectItem

if TYPE_CHECKING:
    from .image_canvas import ImageCanvas


class BandOverlayItem(QGraphicsRectItem):
    """Interactive rectangle for a detected band on the canvas."""

    def __init__(
        self,
        rect: QRectF,
        band: object,
        color: QColor,
        callback,
        remove_callback=None,
    ) -> None:
        super().__init__(rect)
        self.band = band
        self.base_color = color
        self.callback = callback
        self.remove_callback = remove_callback
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self._comparison_slot: str | None = None
        self._update_style()

    def set_comparison_slot(self, slot: str | None) -> None:
        self._comparison_slot = slot
        self._update_style()

    def _update_style(self) -> None:
        slot = getattr(self, "_comparison_slot", None)
        if slot is not None and slot != "":
            if slot == "A":
                hex_color = "#f85149"
            elif slot == "B":
                hex_color = "#58a6ff"
            elif slot.startswith("#"):
                hex_color = slot
            else:
                hex_color = "#2dccb8"
            c = QColor(hex_color)
            self.setPen(QPen(c, 3))
            bg = QColor(c)
            bg.setAlpha(70)
            self.setBrush(bg)
            return

        if self.band.selected:
            c = QColor(self.base_color)
            c.setAlpha(180)
            self.setPen(QPen(c, 1))
            bg = QColor(c)
            bg.setAlpha(40)
            self.setBrush(bg)
        else:
            c = QColor(Colors.FG_SECONDARY)
            c.setAlpha(100)
            self.setPen(QPen(c, 1, Qt.PenStyle.DashLine))
            self.setBrush(QBrush())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.callback:
                self.callback(self.band)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            if self.remove_callback:
                self.remove_callback(self.band)
            event.accept()
        else:
            super().mousePressEvent(event)


class LaneBorderItem(QGraphicsLineItem):
    """Draggable vertical line representing a lane boundary."""

    GRAB_TOLERANCE = 8

    def __init__(
        self,
        border_index: int,
        x: float,
        y_top: float,
        y_bottom: float,
        canvas: ImageCanvas,
    ) -> None:
        super().__init__(x, y_top, x, y_bottom)
        self.border_index = border_index
        self._canvas = canvas
        self._dragging = False
        self._img_width = 0.0

        self._apply_normal_style()
        self.setAcceptHoverEvents(True)
        self.setZValue(20)

    def _apply_normal_style(self) -> None:
        pen = QPen(QColor(Colors.ACCENT_PRIMARY), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)

    def _apply_hover_style(self) -> None:
        pen = QPen(QColor(Colors.ACCENT_PRIMARY_HOVER), 3, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        self.setPen(pen)

    def hoverEnterEvent(self, event) -> None:
        self._apply_hover_style()
        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self._dragging:
            self._apply_normal_style()
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._apply_hover_style()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            new_x = event.scenePos().x()
            margin = 4.0
            new_x = max(margin, min(self._img_width - margin, new_x))
            line = self.line()
            self.setLine(new_x, line.y1(), new_x, line.y2())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._apply_hover_style()
            final_x = self.line().x1()
            self._canvas.lane_border_changed.emit(self.border_index, float(final_x))
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ResizableCropItem(QGraphicsRectItem):
    """A crop rectangle with 8 draggable handles for resizing."""

    HANDLE_SIZE = 10
    _TOP = {0, 1, 2}
    _BOTTOM = {5, 6, 7}
    _LEFT = {0, 3, 5}
    _RIGHT = {2, 4, 7}

    _CURSORS = [
        Qt.CursorShape.SizeFDiagCursor,  # 0 TL
        Qt.CursorShape.SizeVerCursor,  # 1 TC
        Qt.CursorShape.SizeBDiagCursor,  # 2 TR
        Qt.CursorShape.SizeHorCursor,  # 3 ML
        Qt.CursorShape.SizeHorCursor,  # 4 MR
        Qt.CursorShape.SizeBDiagCursor,  # 5 BL
        Qt.CursorShape.SizeVerCursor,  # 6 BC
        Qt.CursorShape.SizeFDiagCursor,  # 7 BR
    ]

    def __init__(self, rect: QRectF, img_rect: QRectF) -> None:
        super().__init__(rect)
        self._img_rect = img_rect
        self._drag_handle: int | None = None
        self._drag_start: QPointF | None = None
        self._rect_at_drag: QRectF | None = None
        self._moving = False

        color = QColor(Colors.ACCENT_PRIMARY)
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        fill = QColor(color)
        fill.setAlpha(25)
        self.setPen(pen)
        self.setBrush(fill)
        self.setZValue(15)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def _handle_rects(self) -> list[QRectF]:
        r = self.rect()
        s = self.HANDLE_SIZE
        cx, cy = r.center().x(), r.center().y()
        return [
            QRectF(r.left() - s / 2, r.top() - s / 2, s, s),
            QRectF(cx - s / 2, r.top() - s / 2, s, s),
            QRectF(r.right() - s / 2, r.top() - s / 2, s, s),
            QRectF(r.left() - s / 2, cy - s / 2, s, s),
            QRectF(r.right() - s / 2, cy - s / 2, s, s),
            QRectF(r.left() - s / 2, r.bottom() - s / 2, s, s),
            QRectF(cx - s / 2, r.bottom() - s / 2, s, s),
            QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
        ]

    def _hit_handle(self, pos: QPointF) -> int | None:
        for i, hr in enumerate(self._handle_rects()):
            if hr.contains(pos):
                return i
        return None

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        painter.save()
        hc = QColor(Colors.ACCENT_PRIMARY)
        painter.setPen(QPen(hc, 1))
        painter.setBrush(QBrush(hc))
        for hr in self._handle_rects():
            painter.drawEllipse(hr)
        painter.restore()

    def boundingRect(self) -> QRectF:
        s = self.HANDLE_SIZE
        return self.rect().adjusted(-s, -s, s, s)

    def hoverMoveEvent(self, event) -> None:
        h = self._hit_handle(event.pos())
        if h is not None:
            self.setCursor(QCursor(self._CURSORS[h]))
        else:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.scenePos()
            self._rect_at_drag = QRectF(self.rect())
            h = self._hit_handle(event.pos())
            if h is not None:
                self._drag_handle = h
                self._moving = False
            else:
                self._drag_handle = None
                self._moving = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        assert self._rect_at_drag is not None  # set together with _drag_start

        delta = event.scenePos() - self._drag_start
        r = QRectF(self._rect_at_drag)
        ir = self._img_rect
        MIN = 20.0

        if self._moving:
            r.translate(delta)
            if r.left() < ir.left():
                r.moveLeft(ir.left())
            if r.top() < ir.top():
                r.moveTop(ir.top())
            if r.right() > ir.right():
                r.moveRight(ir.right())
            if r.bottom() > ir.bottom():
                r.moveBottom(ir.bottom())
        else:
            h = self._drag_handle
            if h in self._TOP:
                r.setTop(max(ir.top(), min(r.bottom() - MIN, r.top() + delta.y())))
            if h in self._BOTTOM:
                r.setBottom(min(ir.bottom(), max(r.top() + MIN, r.bottom() + delta.y())))
            if h in self._LEFT:
                r.setLeft(max(ir.left(), min(r.right() - MIN, r.left() + delta.x())))
            if h in self._RIGHT:
                r.setRight(min(ir.right(), max(r.left() + MIN, r.right() + delta.x())))

        self.setRect(r)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self._rect_at_drag = None
            self._drag_handle = None
            self._moving = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
