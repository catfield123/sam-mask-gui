"""Rendering logic for the image viewer widget."""


import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap

from src.gui.widgets.image_viewer.coordinate_mapper import CoordinateMapper
from src.models.keypoint import KeypointType


class ViewerRenderer:
    """Stateless renderer that paints the image viewer contents.

    All drawing is performed via :meth:`paint`, which reads the current
    viewer state and draws into a ``QPainter``.
    """

    def paint(self, viewer, painter: QPainter):
        """Perform the full paint for a single frame.

        Args:
            - viewer: The ``ImageViewerWidget`` instance (used to read state).
            - painter (QPainter): Active painter targeting the widget.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if viewer.image is None:
            self._paint_placeholder(viewer, painter)
            return

        img_h, img_w = viewer.image.shape[:2]
        display_w = int(img_w * viewer.display_scale)
        display_h = int(img_h * viewer.display_scale)

        scaled_pixmap = self._image_to_scaled_pixmap(viewer.image, display_w, display_h)

        viewer.actual_display_w = scaled_pixmap.width()
        viewer.actual_display_h = scaled_pixmap.height()

        alt_pressed = viewer.is_alt_pressed()

        if alt_pressed and viewer.mask is not None:
            self._paint_alt_preview(viewer, painter, display_w, display_h, img_h, img_w)
        else:
            self._paint_normal(viewer, painter, scaled_pixmap, display_w, display_h, img_h, img_w)

        if not alt_pressed and not viewer.is_brush_mode_active():
            self._paint_keypoints(viewer, painter)

        self._paint_last_keypoint_info(viewer, painter)
        self._paint_center_brush_preview(viewer, painter)
        self._paint_brush_cursor(viewer, painter)
        self._paint_save_indicator(viewer, painter)

    # ------------------------------------------------------------------
    # Sub-renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _paint_placeholder(viewer, painter: QPainter):
        """Draw the empty-state background when no image is loaded."""
        painter.fillRect(viewer.rect(), QColor(50, 50, 50))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(viewer.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")

    @staticmethod
    def _image_to_scaled_pixmap(image: np.ndarray, display_w: int, display_h: int) -> QPixmap:
        """Convert a numpy image to a scaled QPixmap."""
        img_h, img_w = image.shape[:2]
        if len(image.shape) == 3:
            qimage = QImage(image.data, img_w, img_h, img_w * 3, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(image.data, img_w, img_h, img_w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimage)
        return pixmap.scaled(
            display_w,
            display_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _paint_alt_preview(self, viewer, painter: QPainter, display_w: int, display_h: int, img_h: int, img_w: int):
        """Render the Alt-key black-and-white mask preview."""
        mask_h, mask_w = viewer.mask.shape[:2]
        if mask_h != img_h or mask_w != img_w:
            return

        alt_pressed = viewer.is_alt_pressed()
        if viewer._alt_preview_cache is None or viewer._last_alt_state != alt_pressed:
            mask_grayscale = viewer.mask.copy()
            qimage_mask = QImage(
                mask_grayscale.data,
                mask_w,
                mask_h,
                mask_w,
                QImage.Format.Format_Grayscale8,
            )
            pixmap_mask = QPixmap.fromImage(qimage_mask)
            viewer._alt_preview_cache = pixmap_mask.scaled(
                display_w,
                display_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            viewer._last_alt_state = alt_pressed

        painter.fillRect(viewer.rect(), QColor(0, 0, 0))
        painter.drawPixmap(
            viewer.display_offset_x,
            viewer.display_offset_y,
            viewer._alt_preview_cache,
        )

    def _paint_normal(
        self, viewer, painter: QPainter, scaled_pixmap: QPixmap, display_w: int, display_h: int, img_h: int, img_w: int
    ):
        """Render the image with a semi-transparent mask overlay."""
        painter.drawPixmap(viewer.display_offset_x, viewer.display_offset_y, scaled_pixmap)

        if viewer.mask is None:
            return

        mask_h, mask_w = viewer.mask.shape[:2]
        if mask_h != img_h or mask_w != img_w:
            return

        r, g, b = viewer.mask_color
        overlay = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
        overlay[:, :, 0] = r
        overlay[:, :, 1] = g
        overlay[:, :, 2] = b
        overlay[:, :, 3] = (viewer.mask > 0).astype(np.uint8) * viewer.mask_alpha

        qimage_mask = QImage(
            overlay.data,
            mask_w,
            mask_h,
            mask_w * 4,
            QImage.Format.Format_RGBA8888,
        )
        pixmap_mask = QPixmap.fromImage(qimage_mask)
        scaled_mask = pixmap_mask.scaled(
            display_w,
            display_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(viewer.display_offset_x, viewer.display_offset_y, scaled_mask)

    @staticmethod
    def _paint_keypoints(viewer, painter: QPainter):
        """Draw positive (green) and negative (red) keypoint markers."""
        if viewer.image is None:
            return

        img_h, img_w = viewer.image.shape[:2]
        if img_w == 0 or img_h == 0 or viewer.actual_display_w == 0 or viewer.actual_display_h == 0:
            return

        for kp in viewer.keypoints:
            dx, dy = CoordinateMapper.image_to_display(
                kp.x,
                kp.y,
                img_w,
                img_h,
                viewer.display_offset_x,
                viewer.display_offset_y,
                viewer.actual_display_w,
                viewer.actual_display_h,
            )
            color = QColor(0, 255, 0) if kp.type == KeypointType.POSITIVE else QColor(255, 0, 0)
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawEllipse(dx - 5, dy - 5, 10, 10)

    @staticmethod
    def _paint_last_keypoint_info(viewer, painter: QPainter):
        """Draw the info bar showing last keypoint coordinates."""
        if viewer.image is None or viewer.last_keypoint is None:
            return

        img_h, img_w = viewer.image.shape[:2]
        kp = viewer.last_keypoint
        label = "POSITIVE" if kp.type == KeypointType.POSITIVE else "NEGATIVE"
        coord_text = (
            f"Last keypoint: ({kp.x}, {kp.y}) | "
            f"Image size: {img_w}x{img_h} | "
            f"Display: {viewer.actual_display_w}x{viewer.actual_display_h} "
            f"[{label}]"
        )

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(coord_text)
        text_rect.adjust(-5, -2, 5, 2)
        text_rect.moveTopLeft(QPoint(10, 10))

        painter.fillRect(text_rect, QColor(0, 0, 0, 200))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            coord_text,
        )

    @staticmethod
    def _paint_center_brush_preview(viewer, painter: QPainter):
        """Draw the brush size preview circle in the image centre (slider drag)."""
        if not viewer._show_center_brush_preview:
            return
        if viewer.image is None or viewer.actual_display_w == 0:
            return

        img_h, img_w = viewer.image.shape[:2]
        cx_disp, cy_disp = CoordinateMapper.image_to_display(
            img_w // 2,
            img_h // 2,
            img_w,
            img_h,
            viewer.display_offset_x,
            viewer.display_offset_y,
            viewer.actual_display_w,
            viewer.actual_display_h,
        )
        brush_r = viewer.brush_size // 2

        old_mode = painter.compositionMode()
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Difference)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(cx_disp - brush_r, cy_disp - brush_r, brush_r * 2, brush_r * 2)
        painter.setCompositionMode(old_mode)

        size_text = f"Brush: {viewer.brush_size}px"
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tr = fm.boundingRect(size_text)
        tr.adjust(-3, -1, 3, 1)
        tr.moveTopLeft(QPoint(cx_disp + brush_r + 5, cy_disp - tr.height() // 2))
        painter.fillRect(tr, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(tr, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, size_text)

    @staticmethod
    def _paint_brush_cursor(viewer, painter: QPainter):
        """Draw the brush circle that follows the cursor in brush mode."""
        if not viewer.is_brush_mode_active() or viewer.image is None:
            return

        live = viewer.mapFromGlobal(QCursor.pos())
        if not viewer.rect().contains(live):
            return

        coords = CoordinateMapper.widget_to_image_clamped(
            live.x(),
            live.y(),
            viewer.display_offset_x,
            viewer.display_offset_y,
            viewer.actual_display_w,
            viewer.actual_display_h,
            viewer.image.shape[1],
            viewer.image.shape[0],
        )
        if coords is None:
            return

        img_x, img_y = coords
        img_h, img_w = viewer.image.shape[:2]
        dx, dy = CoordinateMapper.image_to_display(
            img_x,
            img_y,
            img_w,
            img_h,
            viewer.display_offset_x,
            viewer.display_offset_y,
            viewer.actual_display_w,
            viewer.actual_display_h,
        )
        brush_r = viewer.brush_size // 2

        old_mode = painter.compositionMode()
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Difference)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(dx - brush_r, dy - brush_r, brush_r * 2, brush_r * 2)
        painter.setCompositionMode(old_mode)

        mode_text = " [INV]" if viewer.is_brush_inverted() else ""
        size_text = f"Brush: {viewer.brush_size}px{mode_text}"
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tr = fm.boundingRect(size_text)
        tr.adjust(-3, -1, 3, 1)
        tr.moveTopLeft(QPoint(dx + brush_r + 5, dy - tr.height() // 2))
        painter.fillRect(tr, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(tr, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, size_text)

    @staticmethod
    def _paint_save_indicator(viewer, painter: QPainter):
        """Draw the saved/unsaved badge near the top-right corner of the image."""
        if viewer.image is None:
            return

        status_text = "Unsaved" if viewer.has_unsaved_changes else "Saved"
        status_color = QColor(255, 0, 0) if viewer.has_unsaved_changes else QColor(0, 255, 0)

        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tr = fm.boundingRect(status_text)
        tr.adjust(-8, -4, 8, 4)

        status_x = viewer.display_offset_x + viewer.actual_display_w - tr.width() - 10
        status_y = max(5, viewer.display_offset_y - tr.height() - 6)
        tr.moveTopLeft(QPoint(status_x, status_y))

        painter.fillRect(tr, QColor(0, 0, 0, 200))
        painter.setPen(status_color)
        painter.drawText(tr, Qt.AlignmentFlag.AlignCenter, status_text)
