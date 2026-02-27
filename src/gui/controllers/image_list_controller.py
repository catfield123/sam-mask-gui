"""Image list management: loading, sorting, thumbnails, and selection."""

from pathlib import Path
from typing import Dict

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox

from src.models import ImageState
from src.services import ImageService


class ImageListController:
    """Manages the image list widget: loading files, lazy thumbnail
    scheduling, sorting, counter updates, and image selection.

    Args:
        - image_list: ``QListWidget`` displaying image entries.
        - mask_counter_label: ``QLabel`` showing the ``X/Y`` counter.
        - sort_combo: ``QComboBox`` for the sort mode.
        - thumbnail_worker: ``ThumbnailLoaderWorker`` background thread.
        - get_state (callable): Returns ``(images_dir, save_dir, image_states,
          mask_service)`` from the main window.
    """

    def __init__(self, image_list, mask_counter_label, sort_combo, thumbnail_worker, get_state):
        """Initialise the image list controller.

        Args:
            - image_list: ``QListWidget`` displaying image entries.
            - mask_counter_label: ``QLabel`` showing the ``X/Y`` counter.
            - sort_combo: ``QComboBox`` for the sort mode.
            - thumbnail_worker: ``ThumbnailLoaderWorker`` background thread.
            - get_state (callable): Returns ``(images_dir, save_dir,
              image_states, mask_service)`` from the main window.
        """
        self._list = image_list
        self._counter_label = mask_counter_label
        self._sort_combo = sort_combo
        self._worker = thumbnail_worker
        self._get_state = get_state

        # Thumbnail caches
        self.thumbnail_cache: Dict[Path, QPixmap] = {}
        self.pending_thumbnail_timers: Dict[Path, QTimer] = {}

        # Debounce timer for scroll events
        self._scroll_timer = QTimer()
        self._scroll_timer.timeout.connect(self.schedule_thumbnail_loading)
        self._scroll_timer.setSingleShot(True)

        # Wire scroll signals
        self._list.verticalScrollBar().valueChanged.connect(self.on_scroll)
        self._list.verticalScrollBar().rangeChanged.connect(self.on_scroll)

    # ------------------------------------------------------------------
    # Loading images
    # ------------------------------------------------------------------

    def load_images(self, on_image_selected_cb, parent_widget=None):
        """Scan the images directory, populate the list, and select the first entry.

        Args:
            - on_image_selected_cb (callable): Slot for ``currentItemChanged``.
            - parent_widget (QWidget | None): Parent for info/error dialogs.
        """
        images_dir, save_dir, image_states, mask_service = self._get_state()
        if images_dir is None:
            return

        image_paths = ImageService.find_images(images_dir)
        if not image_paths:
            QMessageBox.information(parent_widget, "No Images", "No images found in selected folder.")
            return

        self._list.clear()
        image_states.clear()
        self.thumbnail_cache.clear()
        self._worker.clear_queue()
        for timer in self.pending_thumbnail_timers.values():
            timer.stop()
        self.pending_thumbnail_timers.clear()

        for img_path in image_paths:
            state = ImageState(path=img_path)

            if save_dir:
                mask_path = ImageService.get_mask_path(img_path, save_dir)
                if mask_path.exists():
                    state.mask_saved = True
                    try:
                        if mask_service:
                            mask_full = mask_service.load_mask(mask_path)
                            if mask_full is not None:
                                state.mask = mask_full
                    except Exception:
                        pass

            image_states[img_path] = state

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, img_path)
            item.setText(f"✓ {img_path.name}" if state.mask_saved else img_path.name)
            item.setIcon(QIcon())
            item.setSizeHint(QSize(250, 180))
            self._list.addItem(item)

        self.schedule_thumbnail_loading()

        if hasattr(self, "_sort_combo"):
            self.sort_image_list(self._sort_combo.currentIndex())

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

        self.update_mask_counter()

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sort_image_list(self, sort_index: int):
        """Re-sort the image list according to the selected mode.

        Args:
            - sort_index (int): 0 = by name, 1 = unmasked first, 2 = masked first.
        """
        _, _, image_states, _ = self._get_state()
        if self._list.count() == 0:
            return

        current_item = self._list.currentItem()
        current_path = current_item.data(Qt.ItemDataRole.UserRole) if current_item else None

        items_data = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            img_path = item.data(Qt.ItemDataRole.UserRole)
            if not img_path:
                continue
            state = image_states.get(img_path)
            has_mask = bool(state and (state.mask_saved or state.mask is not None))
            items_data.append(
                (
                    img_path,
                    has_mask,
                    item.text(),
                    item.icon(),
                    item.sizeHint(),
                )
            )

        if sort_index == 0:
            items_data.sort(key=lambda x: x[0].name.lower())
        elif sort_index == 1:
            items_data.sort(key=lambda x: (x[1], x[0].name.lower()))
        elif sort_index == 2:
            items_data.sort(key=lambda x: (not x[1], x[0].name.lower()))

        self._list.clear()
        for img_path, _has_mask, text, icon, size_hint in items_data:
            new_item = QListWidgetItem()
            new_item.setData(Qt.ItemDataRole.UserRole, img_path)
            new_item.setText(text)
            new_item.setIcon(icon)
            new_item.setSizeHint(size_hint)
            self._list.addItem(new_item)

        if current_path:
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == current_path:
                    self._list.setCurrentItem(item)
                    break

    # ------------------------------------------------------------------
    # List / counter helpers
    # ------------------------------------------------------------------

    def update_image_list(self):
        """Refresh list item labels (e.g. saved-indicator checkmarks)."""
        _, _, image_states, _ = self._get_state()
        for i in range(self._list.count()):
            item = self._list.item(i)
            img_path = item.data(Qt.ItemDataRole.UserRole)
            if not img_path:
                continue
            state = image_states.get(img_path)
            if state is None:
                continue
            is_currently_saved = state.mask_saved and (state.state_version == 0 or not state.has_unsaved_changes)
            if is_currently_saved:
                if not item.text().startswith("✓"):
                    item.setText(f"✓ {img_path.name}")
            else:
                if item.text().startswith("✓"):
                    item.setText(img_path.name)
        self.update_mask_counter()

    def update_mask_counter(self):
        """Recompute and display the ``X/Y`` mask counter."""
        _, _, image_states, _ = self._get_state()
        total = len(image_states)
        if total == 0:
            self._counter_label.setText("0/0")
            return
        with_mask = sum(1 for s in image_states.values() if s.mask_saved or s.mask is not None)
        self._counter_label.setText(f"{with_mask}/{total}")

    # ------------------------------------------------------------------
    # Thumbnail scheduling
    # ------------------------------------------------------------------

    def on_scroll(self):
        """Debounce scroll events before scheduling thumbnail loads."""
        self._scroll_timer.stop()
        self._scroll_timer.start(100)

    def schedule_thumbnail_loading(self):
        """Queue visible and nearby thumbnails for background loading."""
        if self._list.count() == 0:
            return

        viewport = self._list.viewport()
        vp_rect = viewport.rect()
        first = self._list.indexAt(vp_rect.topLeft())
        last = self._list.indexAt(vp_rect.bottomLeft())
        if not first.isValid():
            return

        start = first.row()
        end = last.row() if last.isValid() else self._list.count() - 1

        visible_paths = []
        for row in range(start, end + 1):
            item = self._list.item(row)
            if item is None:
                continue
            img_path = item.data(Qt.ItemDataRole.UserRole)
            if img_path and (item.icon().isNull() or img_path not in self.thumbnail_cache):
                visible_paths.append(img_path)

        buf = max(5, end - start + 1)
        buf_start = max(0, start - buf)
        buf_end = min(self._list.count() - 1, end + buf)

        buffer_paths = []
        for row in range(buf_start, buf_end + 1):
            if start <= row <= end:
                continue
            item = self._list.item(row)
            if item is None:
                continue
            img_path = item.data(Qt.ItemDataRole.UserRole)
            if img_path and (item.icon().isNull() or img_path not in self.thumbnail_cache):
                buffer_paths.append(img_path)

        for img_path in visible_paths:
            if img_path in self.pending_thumbnail_timers:
                self.pending_thumbnail_timers[img_path].stop()
                del self.pending_thumbnail_timers[img_path]

            timer = QTimer()
            timer.setSingleShot(True)

            def _make_cb(p):
                """Create a closure that captures the image path."""
                return lambda: self._load_after_delay(p, 1)

            timer.timeout.connect(_make_cb(img_path))
            timer.start(300)
            self.pending_thumbnail_timers[img_path] = timer

        if buffer_paths:
            self._worker.add_batch_to_queue(buffer_paths, priority=2)

    def on_thumbnail_loaded(self, img_path: Path, thumbnail: QPixmap):
        """Handle a thumbnail delivered by the background worker.

        Args:
            - img_path (Path): The image the thumbnail belongs to.
            - thumbnail (QPixmap): The rendered thumbnail.
        """
        self.thumbnail_cache[img_path] = thumbnail
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == img_path:
                item.setIcon(QIcon(thumbnail))
                break

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def on_image_selected(self, current: QListWidgetItem, load_image_cb):
        """Handle the user clicking a different image in the list.

        Args:
            - current (QListWidgetItem | None): Newly selected item.
            - load_image_cb (callable): Callback to load the selected image
              into the viewer and predictor.

        Returns:
            - Path | None: The selected image path, or ``None``.
        """
        if current is None:
            return None

        img_path = current.data(Qt.ItemDataRole.UserRole)
        if not img_path:
            return None

        if img_path in self.pending_thumbnail_timers:
            self.pending_thumbnail_timers[img_path].stop()
            del self.pending_thumbnail_timers[img_path]

        if img_path not in self.thumbnail_cache:
            self._worker.add_to_queue(img_path, priority=0)

        load_image_cb(img_path)
        return img_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_after_delay(self, img_path: Path, priority: int):
        """Enqueue a thumbnail if its item is still visible after the delay.

        Args:
            - img_path (Path): Image to check.
            - priority (int): Queue priority for the worker.
        """
        if img_path in self.pending_thumbnail_timers:
            del self.pending_thumbnail_timers[img_path]
        if self._is_visible(img_path):
            self._worker.add_to_queue(img_path, priority=priority)

    def _is_visible(self, img_path: Path) -> bool:
        """Return ``True`` if *img_path* is currently inside the visible viewport."""
        vp_rect = self._list.viewport().rect()
        first = self._list.indexAt(vp_rect.topLeft())
        last = self._list.indexAt(vp_rect.bottomLeft())
        if not first.isValid():
            return False
        start = first.row()
        end = last.row() if last.isValid() else self._list.count() - 1
        for row in range(start, end + 1):
            item = self._list.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == img_path:
                return True
        return False
