"""Core image viewer widget with zoom, pan, brush, and keypoint support."""

from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from src.gui.widgets.image_viewer.brush_engine import BrushEngine
from src.gui.widgets.image_viewer.coordinate_mapper import CoordinateMapper
from src.gui.widgets.image_viewer.renderer import ViewerRenderer
from src.models.keypoint import Keypoint, KeypointType


class ImageViewerWidget(QWidget):
    """Interactive image viewer supporting keypoint placement, brush painting,
    zoom/pan, and mask overlay.

    Signals:
        - keypoint_added(int, int, int): ``(x, y, keypoint_type)`` in image space.
        - brush_stroke_started(): Emitted at the beginning of a brush stroke.
        - brush_stroke_finished(np.ndarray): Emitted with the final mask.
        - zoom_changed(float): New zoom factor.
        - brush_size_changed(int): New brush diameter in screen pixels.
        - brush_max_changed(int): Updated maximum brush size.
    """

    keypoint_added = pyqtSignal(int, int, int)
    brush_stroke_started = pyqtSignal()
    brush_stroke_finished = pyqtSignal(np.ndarray)
    zoom_changed = pyqtSignal(float)
    brush_size_changed = pyqtSignal(int)
    brush_max_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialise the image viewer widget.

        Args:
            - parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)

        # Image / mask state
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None  # type: ignore[assignment]
        self.keypoints: List[Keypoint] = []
        self.last_keypoint: Optional[Keypoint] = None
        self.has_unsaved_changes: bool = False

        # Display geometry
        self.display_scale: float = 1.0
        self.base_display_scale: float = 1.0
        self.display_offset_x: int = 0
        self.display_offset_y: int = 0
        self.actual_display_w: int = 0
        self.actual_display_h: int = 0

        # Zoom / pan
        self.zoom_factor: float = 1.0
        self._pan_offset_x: int = 0
        self._pan_offset_y: int = 0
        self._is_panning: bool = False
        self._pan_start_pos: Optional[QPoint] = None
        self._pan_start_offset: Optional[Tuple[int, int]] = None

        # Brush
        self.brush_size: int = 40
        self.current_mouse_pos: Optional[QPoint] = None
        self._brush = BrushEngine()

        # Mask appearance
        self.mask_color: Tuple[int, int, int] = (255, 0, 0)
        self.mask_alpha: int = 128

        # Alt-preview cache
        self._alt_preview_cache: Optional[object] = None
        self._last_alt_state: bool = False
        self._alt_key_pressed: bool = False

        # Centre brush preview (shown while slider is held)
        self._show_center_brush_preview: bool = False
        self._last_max_brush: int = 2000

        # Renderer
        self._renderer = ViewerRenderer()

        # Widget config
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        """Replace the displayed image and reset zoom/pan.

        Args:
            - image (np.ndarray): RGB image array ``(H, W, 3)``.
        """
        self.image = image
        self.zoom_factor = 1.0
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        if self._brush.is_drawing:
            self._finalize_brush_stroke()
        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def set_mask(self, mask: Optional[np.ndarray]):
        """Set or clear the mask overlay.

        Args:
            - mask (np.ndarray | None): Grayscale mask, or ``None`` to clear.
        """
        self.mask = mask
        self._alt_preview_cache = None
        self._update_display()

    def set_unsaved_changes(self, has_unsaved_changes: bool):
        """Update the unsaved-changes indicator.

        Args:
            - has_unsaved_changes (bool): Whether there are unsaved changes.
        """
        self.has_unsaved_changes = has_unsaved_changes
        self.update()

    def set_keypoints(self, keypoints: List[Keypoint]):
        """Replace the displayed keypoints.

        Args:
            - keypoints (list[Keypoint]): New keypoint list.
        """
        self.keypoints = keypoints
        self.last_keypoint = keypoints[-1] if keypoints else None
        self._update_display()

    def set_zoom(self, zoom_factor: float):
        """Set the zoom level (1.0 = fit-to-widget).

        Args:
            - zoom_factor (float): Desired zoom level (clamped to [1.0, 5.0]).
        """
        if self.image is None:
            return

        img_h, img_w = self.image.shape[:2]
        widget_size = self.size()
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(5.0, zoom_factor))

        if self.zoom_factor <= 1.0:
            self._pan_offset_x = 0
            self._pan_offset_y = 0
            self._update_display()
            self.zoom_changed.emit(self.zoom_factor)
            return

        # Zoom towards widget centre
        cx = widget_size.width() / 2.0
        cy = widget_size.height() / 2.0
        old_scale = self.base_display_scale * old_zoom
        if old_scale != 0:
            img_x = (cx - self.display_offset_x) / old_scale
            img_y = (cy - self.display_offset_y) / old_scale
        else:
            img_x, img_y = img_w / 2.0, img_h / 2.0

        new_scale = self.base_display_scale * self.zoom_factor
        new_w = int(img_w * new_scale)
        new_h = int(img_h * new_scale)
        self._pan_offset_x = int(cx - img_x * new_scale - (widget_size.width() - new_w) // 2)
        self._pan_offset_y = int(cy - img_y * new_scale - (widget_size.height() - new_h) // 2)

        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def set_brush_size(self, size: int):
        """Set the brush diameter in screen pixels.

        Args:
            - size (int): Diameter (>= 2).
        """
        self.brush_size = max(2, size)
        self.update()
        self.brush_size_changed.emit(self.brush_size)

    def set_mask_color(self, r: int, g: int, b: int):
        """Set the mask overlay colour.

        Args:
            - r (int): Red channel (0-255).
            - g (int): Green channel (0-255).
            - b (int): Blue channel (0-255).
        """
        self.mask_color = (r, g, b)
        self._alt_preview_cache = None
        self.update()

    def set_mask_alpha(self, alpha: int):
        """Set the mask overlay opacity.

        Args:
            - alpha (int): Opacity value (clamped to [20, 255]).
        """
        self.mask_alpha = max(20, min(255, alpha))
        self._alt_preview_cache = None
        self.update()

    def show_center_brush_preview(self, show: bool):
        """Toggle the centred brush-size preview (used while dragging the slider).

        Args:
            - show (bool): Whether to display the preview.
        """
        self._show_center_brush_preview = show
        self.update()

    # ------------------------------------------------------------------
    # Mode queries
    # ------------------------------------------------------------------

    def is_brush_mode_active(self) -> bool:
        """Return ``True`` when Shift is held (brush mode)."""
        return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

    def is_brush_inverted(self) -> bool:
        """Return ``True`` when Ctrl+Shift is held (inverted brush)."""
        mods = QApplication.keyboardModifiers()
        return bool(mods & Qt.KeyboardModifier.ShiftModifier) and bool(mods & Qt.KeyboardModifier.ControlModifier)

    def is_alt_pressed(self) -> bool:
        """Return ``True`` when Alt is held (mask-only preview)."""
        return self._alt_key_pressed

    # ------------------------------------------------------------------
    # Display geometry
    # ------------------------------------------------------------------

    def _update_display(self):
        """Recalculate display scale, offsets, and clamp pan bounds."""
        if self.image is None:
            self.update()
            return

        widget_size = self.size()
        img_h, img_w = self.image.shape[:2]

        scale_x = widget_size.width() / img_w
        scale_y = widget_size.height() / img_h
        self.base_display_scale = min(scale_x, scale_y)
        self.display_scale = self.base_display_scale * self.zoom_factor

        display_w = int(img_w * self.display_scale)
        display_h = int(img_h * self.display_scale)
        ww, wh = widget_size.width(), widget_size.height()

        # Clamp pan offsets so the image edges stay within the widget
        if display_w > ww:
            cx = (ww - display_w) // 2
            self._pan_offset_x = max(ww - display_w - cx, min(-cx, self._pan_offset_x))
        else:
            self._pan_offset_x = 0

        if display_h > wh:
            cy = (wh - display_h) // 2
            self._pan_offset_y = max(wh - display_h - cy, min(-cy, self._pan_offset_y))
        else:
            self._pan_offset_y = 0

        self.display_offset_x = (ww - display_w) // 2 + self._pan_offset_x
        self.display_offset_y = (wh - display_h) // 2 + self._pan_offset_y

        # Update maximum brush size
        new_max = max(2, min(min(display_w, display_h), min(ww, wh)))
        if new_max != self._last_max_brush:
            self._last_max_brush = new_max
            self.brush_max_changed.emit(new_max)
        if self.brush_size > new_max:
            self.brush_size = new_max
            self.brush_size_changed.emit(self.brush_size)

        self.update()

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        """Recalculate display geometry on widget resize."""
        super().resizeEvent(event)
        self._update_display()

    def focusOutEvent(self, event):
        """Finalize any in-progress brush stroke when focus is lost."""
        super().focusOutEvent(event)
        if self._brush.is_drawing:
            self._finalize_brush_stroke()
        self.update()

    def paintEvent(self, event):
        """Delegate all rendering to :class:`ViewerRenderer`."""
        painter = QPainter(self)
        self._renderer.paint(self, painter)

    # -- Keyboard -------------------------------------------------------

    def keyPressEvent(self, event):
        """Handle key-down: Shift toggles brush mode, Alt toggles mask preview."""
        if event.key() == Qt.Key.Key_Shift:
            self.setCursor(Qt.CursorShape.BlankCursor)
            self.update()
        elif event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            if not self._alt_key_pressed:
                self._alt_preview_cache = None
                self._alt_key_pressed = True
                self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key-up: finalize brush on Shift release, clear Alt preview."""
        if event.key() == Qt.Key.Key_Shift:
            if self._brush.is_drawing:
                self._finalize_brush_stroke()
            self.unsetCursor()
            self.current_mouse_pos = None
            self.update()
        elif event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat():
            if self._alt_key_pressed:
                self._alt_preview_cache = None
                self._alt_key_pressed = False
                self.update()
        super().keyReleaseEvent(event)

    def eventFilter(self, obj, event):
        """Global filter to track Alt reliably on Linux (avoids menu-bar steal)."""
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Alt and not self._alt_key_pressed:
                self._alt_preview_cache = None
                self._alt_key_pressed = True
                self.update()
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Alt and self._alt_key_pressed:
                self._alt_preview_cache = None
                self._alt_key_pressed = False
                self.update()
        return super().eventFilter(obj, event)

    def hideEvent(self, event):
        """Remove the global event filter when the widget is hidden."""
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._alt_key_pressed:
            self._alt_preview_cache = None
            self._alt_key_pressed = False
        super().hideEvent(event)

    def showEvent(self, event):
        """Re-install the global event filter when the widget is shown."""
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        super().showEvent(event)

    # -- Mouse ----------------------------------------------------------

    def mousePressEvent(self, event):
        """Route mouse clicks to panning, brush, or keypoint logic."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self._pan_start_offset = (self._pan_offset_x, self._pan_offset_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.image is None or self.actual_display_w == 0 or self.actual_display_h == 0:
            return

        wx, wy = event.position().x(), event.position().y()
        if not (
            self.display_offset_x <= wx < self.display_offset_x + self.actual_display_w
            and self.display_offset_y <= wy < self.display_offset_y + self.actual_display_h
        ):
            return

        img_h, img_w = self.image.shape[:2]
        coords = CoordinateMapper.widget_to_image_clamped(
            wx,
            wy,
            self.display_offset_x,
            self.display_offset_y,
            self.actual_display_w,
            self.actual_display_h,
            img_w,
            img_h,
        )
        if coords is None:
            return
        img_x, img_y = coords

        if self.is_brush_mode_active():
            self._handle_brush_press(event.button(), img_x, img_y)
        else:
            self._handle_keypoint_press(event.button(), img_x, img_y)

    def mouseMoveEvent(self, event):
        """Handle mouse movement: panning, brush drawing, or cursor update."""
        if self._is_panning and self._pan_start_pos is not None:
            delta = event.position().toPoint() - self._pan_start_pos
            self._pan_offset_x = self._pan_start_offset[0] + delta.x()
            self._pan_offset_y = self._pan_start_offset[1] + delta.y()
            self._update_display()
            return

        self._sync_alt_state()

        # Keep cursor in sync with current mode
        if self.is_brush_mode_active():
            if self.cursor().shape() != Qt.CursorShape.BlankCursor:
                self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            if self.cursor().shape() == Qt.CursorShape.BlankCursor:
                self.unsetCursor()

        if self.is_brush_mode_active() and self._brush.is_drawing:
            self._handle_brush_move(event)
        elif self._brush.is_drawing and not self.is_brush_mode_active():
            self._finalize_brush_stroke()

        self.current_mouse_pos = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        """Finish panning or brush stroke on button release."""
        if event.button() == Qt.MouseButton.MiddleButton and self._is_panning:
            self._is_panning = False
            self.unsetCursor()
            return

        if self._brush.is_drawing:
            self._finalize_brush_stroke()
            self.current_mouse_pos = None
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle scroll: Ctrl+scroll = zoom, Shift/Alt+scroll = brush size."""
        self._sync_alt_state()

        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.angleDelta().x()

        ctrl = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)

        if ctrl and self.image is not None:
            self._handle_zoom_wheel(event, delta)
        elif self.is_brush_mode_active() or self.is_alt_pressed():
            self._handle_brush_size_wheel(delta)
        else:
            super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_alt_state(self):
        """Sync the Alt flag with actual OS modifier state (Linux fallback)."""
        actual = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        if actual != self._alt_key_pressed:
            if not actual:
                self._alt_preview_cache = None
            self._alt_key_pressed = actual
            self.update()

    def _handle_keypoint_press(self, button, img_x: int, img_y: int):
        """Emit keypoint signals for left/right clicks."""
        if button == Qt.MouseButton.LeftButton:
            self.keypoint_added.emit(img_x, img_y, KeypointType.POSITIVE.value)
        elif button == Qt.MouseButton.RightButton:
            self.keypoint_added.emit(img_x, img_y, KeypointType.NEGATIVE.value)

    def _handle_brush_press(self, button, img_x: int, img_y: int):
        """Start a brush stroke on left/right click in brush mode."""
        if button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return

        inverted = self.is_brush_inverted()
        mode = (0 if inverted else 255) if button == Qt.MouseButton.LeftButton else 255 if inverted else 0

        self.brush_stroke_started.emit()
        radius = CoordinateMapper.brush_image_radius(self.brush_size, self.display_scale)
        assert self.image is not None
        mask = self._brush.start_stroke(img_x, img_y, mode, self.mask, self.image.shape[:2], radius)
        self.set_mask(mask)

    def _handle_brush_move(self, event):
        """Continue the active brush stroke during mouse movement."""
        wx, wy = event.position().x(), event.position().y()
        if self.image is None:
            return

        img_h, img_w = self.image.shape[:2]
        coords = CoordinateMapper.widget_to_image_clamped(
            wx,
            wy,
            self.display_offset_x,
            self.display_offset_y,
            self.actual_display_w,
            self.actual_display_h,
            img_w,
            img_h,
        )
        if coords is None:
            return

        img_x, img_y = coords
        radius = CoordinateMapper.brush_image_radius(self.brush_size, self.display_scale)
        mask = self._brush.continue_stroke(img_x, img_y, radius)
        if mask is not None:
            self.set_mask(mask)

    def _finalize_brush_stroke(self):
        """End the current brush stroke and emit the result."""
        result = self._brush.finalize_stroke()
        if result is not None:
            self.brush_stroke_finished.emit(result)
        if not self.is_brush_mode_active():
            self.unsetCursor()

    def _handle_zoom_wheel(self, event, delta: int):
        """Zoom towards the cursor position on Ctrl+scroll."""
        assert self.image is not None
        mx, my = event.position().x(), event.position().y()
        img_h, img_w = self.image.shape[:2]
        widget_size = self.size()

        if self.display_scale != 0:
            img_x = (mx - self.display_offset_x) / self.display_scale
            img_y = (my - self.display_offset_y) / self.display_scale
        else:
            img_x, img_y = img_w / 2.0, img_h / 2.0

        factor = 1.1 if delta > 0 else 1.0 / 1.1
        self.zoom_factor = max(1.0, min(5.0, self.zoom_factor * factor))

        if self.zoom_factor <= 1.0:
            self._pan_offset_x = 0
            self._pan_offset_y = 0
            self._update_display()
            self.zoom_changed.emit(self.zoom_factor)
            return

        new_scale = self.base_display_scale * self.zoom_factor
        new_w = int(img_w * new_scale)
        new_h = int(img_h * new_scale)
        self._pan_offset_x = int(mx - img_x * new_scale - (widget_size.width() - new_w) // 2)
        self._pan_offset_y = int(my - img_y * new_scale - (widget_size.height() - new_h) // 2)

        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def _handle_brush_size_wheel(self, delta: int):
        """Adjust brush size on scroll in brush/Alt mode."""
        step = max(2, int(self.brush_size * 0.1))
        if self.actual_display_w > 0 and self.width() > 0:
            max_brush = max(
                2,
                min(
                    min(self.actual_display_w, self.actual_display_h),
                    min(self.width(), self.height()),
                ),
            )
        else:
            max_brush = 2000

        if delta > 0:
            self.brush_size = min(max_brush, self.brush_size + step)
        else:
            self.brush_size = max(2, self.brush_size - step)

        self.update()
        self.brush_size_changed.emit(self.brush_size)
