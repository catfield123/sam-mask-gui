"""Background thread for loading a model without blocking the UI."""

from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal

from src.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoadWorker(QThread):
    """Run a model-construction callable in a worker thread.

    Signals:
        - loaded(object): Emitted with the loaded predictor instance.
        - error(str): Emitted when loading fails.
    """

    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, loader: Callable[[], Any], parent=None):
        """Initialise the worker with a model loader callable."""
        super().__init__(parent)
        self._loader = loader
        logger.debug("ModelLoadWorker created")

    def run(self):
        """Execute the loader in the worker thread."""
        logger.debug("ModelLoadWorker run() started")
        try:
            model = self._loader()
            logger.info("ModelLoadWorker: model loaded successfully")
            self.loaded.emit(model)
        except Exception as exc:
            logger.error("ModelLoadWorker: load failed — %s", exc, exc_info=True)
            self.error.emit(str(exc))
