"""Background thread for loading image thumbnails."""

from pathlib import Path
from typing import List, Tuple

import cv2
from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


class ThumbnailLoaderWorker(QThread):
    """Worker thread that loads and resizes thumbnails from a priority queue.

    Priority levels:
        - 0: currently selected item (highest)
        - 1: visible items
        - 2: buffer items around the viewport

    Signals:
        - thumbnail_loaded(Path, QPixmap): Emitted when a thumbnail is ready.
    """

    thumbnail_loaded = pyqtSignal(Path, QPixmap)

    def __init__(self, parent=None):
        """Initialise the thumbnail loader thread.

        Args:
            - parent (QObject | None): Parent object.
        """
        super().__init__(parent)
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.queue: List[Tuple[int, Path]] = []
        self.running = True
        self.thumbnail_size = 200

    def add_to_queue(self, img_path: Path, priority: int = 2):
        """Enqueue a single image path with the given priority.

        Args:
            - img_path (Path): Image to generate a thumbnail for.
            - priority (int): Lower number = higher priority.
        """
        self.mutex.lock()
        self.queue = [(p, path) for p, path in self.queue if path != img_path]
        inserted = False
        for i, (p, _path) in enumerate(self.queue):
            if priority < p:
                self.queue.insert(i, (priority, img_path))
                inserted = True
                break
        if not inserted:
            self.queue.append((priority, img_path))
        self.mutex.unlock()
        self.wait_condition.wakeOne()

    def add_batch_to_queue(self, img_paths: List[Path], priority: int = 2):
        """Enqueue multiple image paths at the same priority.

        Args:
            - img_paths (list[Path]): Images to generate thumbnails for.
            - priority (int): Lower number = higher priority.
        """
        self.mutex.lock()
        existing_paths = {path for _, path in self.queue}
        for img_path in img_paths:
            if img_path in existing_paths:
                self.queue = [(p, path) for p, path in self.queue if path != img_path]
            inserted = False
            for i, (p, _path) in enumerate(self.queue):
                if priority < p:
                    self.queue.insert(i, (priority, img_path))
                    inserted = True
                    break
            if not inserted:
                self.queue.append((priority, img_path))
        self.queue.sort(key=lambda x: x[0])
        self.mutex.unlock()
        self.wait_condition.wakeOne()

    def clear_queue(self):
        """Remove all pending items from the queue."""
        self.mutex.lock()
        self.queue.clear()
        self.mutex.unlock()

    def stop(self):
        """Signal the thread to stop and wait for it to finish."""
        self.mutex.lock()
        self.running = False
        self.queue.clear()
        self.mutex.unlock()
        self.wait_condition.wakeAll()
        self.wait()

    def run(self):
        """Main loop: dequeue items and emit thumbnails."""
        while self.running:
            self.mutex.lock()
            if not self.queue:
                self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
                continue

            _priority, img_path = self.queue.pop(0)
            self.mutex.unlock()

            thumbnail = self._create_thumbnail(img_path)
            if not thumbnail.isNull():
                self.thumbnail_loaded.emit(img_path, thumbnail)

            self.msleep(10)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_thumbnail(self, img_path: Path) -> QPixmap:
        """Read an image and return a resized QPixmap thumbnail.

        Args:
            - img_path (Path): Image file to read.

        Returns:
            - QPixmap: Thumbnail pixmap, or a null pixmap on failure.
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return QPixmap()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            if max(h, w) > self.thumbnail_size:
                scale = self.thumbnail_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            h, w = img_rgb.shape[:2]
            qimage = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception:
            return QPixmap()
