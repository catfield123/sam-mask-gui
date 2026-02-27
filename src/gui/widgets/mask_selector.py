"""Widget for choosing among multiple mask candidates."""

from typing import List

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class MaskSelectorWidget(QWidget):
    """Displays mask candidates sorted by score and emits a signal on selection.

    Signals:
        - mask_selected(int): Emitted with the original candidate index when
          the user clicks a different row.
    """

    mask_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialise the mask selector with an empty list.

        Args:
            - parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("Mask Variants")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        self.mask_list = QListWidget()
        self.mask_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.mask_list)

    def set_masks(self, masks: List[np.ndarray], scores: np.ndarray, image: np.ndarray):
        """Populate the list with mask candidates.

        Args:
            - masks (list[np.ndarray]): Binary mask arrays.
            - scores (np.ndarray): Confidence score per mask.
            - image (np.ndarray): Source image used for preview thumbnails.
        """
        self.mask_list.clear()

        if not masks:
            return

        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            mask = masks[idx]
            score = scores[idx]

            preview = self._create_preview(image, mask, size=150)

            item = QListWidgetItem()
            item.setIcon(QIcon(preview))
            item.setText(f"Score: {score:.3f}")
            item.setData(Qt.ItemDataRole.UserRole, int(idx))
            item.setSizeHint(QSize(200, 160))
            self.mask_list.addItem(item)

        if self.mask_list.count() > 0:
            self.mask_list.setCurrentRow(0)

    def select_mask(self, original_index: int):
        """Select the row corresponding to the original candidate index."""
        for row in range(self.mask_list.count()):
            item = self.mask_list.item(row)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == original_index:
                self.mask_list.setCurrentRow(row)
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_preview(image: np.ndarray, mask: np.ndarray, size: int = 150) -> QPixmap:
        """Render a small preview of *mask* overlaid on *image*.

        Args:
            - image (np.ndarray): Source RGB image.
            - mask (np.ndarray): Binary mask (same spatial dims as *image*).
            - size (int): Maximum side length of the preview thumbnail.

        Returns:
            - QPixmap: The rendered preview, or an empty pixmap on error.
        """
        try:
            img_h, img_w = image.shape[:2]

            if max(img_h, img_w) > size:
                scale = size / max(img_h, img_w)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                img_resized = image
                mask_resized = mask

            if len(img_resized.shape) == 2:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_resized.copy()

            mask_bool = mask_resized > 0
            img_rgb[mask_bool] = img_rgb[mask_bool] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5

            h, w = img_rgb.shape[:2]
            qimage = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception:
            return QPixmap()

    def _on_row_changed(self, row: int):
        """Emit ``mask_selected`` with the original candidate index."""
        if row >= 0:
            item = self.mask_list.item(row)
            if item:
                idx = item.data(Qt.ItemDataRole.UserRole)
                self.mask_selected.emit(int(idx))
