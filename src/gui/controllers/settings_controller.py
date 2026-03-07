"""Configuration persistence and settings dialog integration."""

from PyQt6.QtCore import QTimer

from src.gui.dialogs import SettingsDialog
from src.logging_config import get_logger
from src.services import ConfigService

logger = get_logger(__name__)


class SettingsController:
    """Handles config load/save, folder selection, and the settings dialog.

    Args:
        - config_service (ConfigService): Service for reading/writing the
          JSON config file.
        - get_window_state (callable): Returns a dict with the current
          window-level state keys: ``sam2_checkpoint_path``, ``max_side``,
          ``predictor``, ``mask_service``, ``images_dir``, ``save_dir``.
        - set_window_state (callable): Accepts a dict of state updates to
          apply back to the main window.
    """

    def __init__(self, config_service: ConfigService, get_window_state, set_window_state):
        """Initialise the settings controller.

        Args:
            - config_service (ConfigService): Service for reading/writing
              the JSON config file.
            - get_window_state (callable): Returns a dict of current
              window-level state.
            - set_window_state (callable): Applies a dict of state updates
              back to the main window.
        """
        self._cfg = config_service
        self._get = get_window_state
        self._set = set_window_state
        logger.debug("SettingsController initialised")

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def show_settings(self, parent_widget, load_current_image_cb) -> bool:
        """Open the settings dialog and apply changes.

        Args:
            - parent_widget (QWidget): Dialog parent.
            - load_current_image_cb (callable): Called when the predictor or
              max-side changes and the current image must be reloaded.

        Returns:
            bool: True if the user accepted the dialog and settings were applied.
        """
        logger.debug("show_settings: opening dialog")
        ws = self._get()
        dialog = SettingsDialog(
            parent_widget,
            sam2_checkpoint_path=ws.get("sam2_checkpoint_path", ""),
            sam3_checkpoint_path=ws.get("sam3_checkpoint_path"),
            sam3_bpe_path=ws.get("sam3_bpe_path"),
            max_side=ws.get("max_side", 1024),
        )
        if not dialog.exec():
            logger.debug("show_settings: user cancelled")
            return False

        logger.debug("show_settings: user accepted, applying updates")
        old_ckpt = ws.get("sam2_checkpoint_path", "")
        old_sam3_ckpt = ws.get("sam3_checkpoint_path")
        old_sam3_bpe = ws.get("sam3_bpe_path")
        old_max = ws.get("max_side", 1024)

        new_ckpt = dialog.get_sam2_checkpoint_path()
        new_sam3_ckpt = dialog.get_sam3_checkpoint_path()
        new_sam3_bpe = dialog.get_sam3_bpe_path()
        new_max = dialog.get_max_side()

        updates = {
            "sam2_checkpoint_path": new_ckpt,
            "sam3_checkpoint_path": new_sam3_ckpt,
            "sam3_bpe_path": new_sam3_bpe,
            "max_side": new_max,
        }
        self._set(updates)
        logger.info("Settings applied: checkpoint=%s, sam3_ckpt=%s, max_side=%s", new_ckpt, new_sam3_ckpt, new_max)
        self.save_config()

        # Normalize for comparison: strip paths, treat None and "" as equivalent.
        def _norm(s):
            return (s or "").strip() or None

        ckpt_changed = _norm(old_ckpt) != _norm(new_ckpt)
        sam3_changed = _norm(old_sam3_ckpt) != _norm(new_sam3_ckpt) or _norm(old_sam3_bpe) != _norm(new_sam3_bpe)
        max_changed = old_max != new_max

        ws = self._get()  # re-read after set

        if ckpt_changed or sam3_changed or max_changed:
            logger.debug("Settings changed (ckpt=%s, sam3=%s, max=%s), releasing predictors", ckpt_changed, sam3_changed, max_changed)
            release_predictors_cb = ws.get("release_predictors_cb")
            if callable(release_predictors_cb):
                release_predictors_cb()
            self._set(
                {
                    "predictor": None,
                    "sam3_predictor": None,
                    "mask_service": None,
                }
            )
            if self._get().get("current_image_path"):
                load_current_image_cb()
        return True

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def load_config(self, load_images_cb, check_masks_cb):
        """Read the config file and restore application state.

        Args:
            - load_images_cb (callable): Called (deferred) to populate the image list.
            - check_masks_cb (callable): Called (deferred) to scan for saved masks.
        """
        logger.debug("load_config: loading from config service")
        config = self._cfg.load()
        if not config:
            logger.debug("load_config: no config, skipping restore")
            return

        ckpt = self._cfg.get_sam2_checkpoint_path(config)
        sam3_ckpt = self._cfg.get_sam3_checkpoint_path(config)
        sam3_bpe = self._cfg.get_sam3_bpe_path(config)
        max_side = self._cfg.get_max_side(config)

        updates = {
            "sam2_checkpoint_path": ckpt or "",
            "sam3_checkpoint_path": sam3_ckpt,
            "sam3_bpe_path": sam3_bpe,
            "keep_models_loaded": self._cfg.get_keep_models_loaded(config),
            "max_side": max_side,
        }

        images_dir = self._cfg.get_images_dir(config)
        updates["images_dir"] = images_dir

        save_dir = self._cfg.get_save_dir(config)
        updates["save_dir"] = save_dir

        self._set(updates)
        logger.info("load_config: restored state (ckpt=%s, sam3_ckpt=%s, images_dir=%s, save_dir=%s)", ckpt, sam3_ckpt, images_dir, save_dir)

        if images_dir:
            QTimer.singleShot(0, load_images_cb)
        if images_dir and save_dir:
            QTimer.singleShot(100, check_masks_cb)

    def save_config(self) -> None:
        """Persist the current application state to the config file."""
        logger.debug("save_config: persisting state")
        ws = self._get()
        self._cfg.save(
            {
                "sam2_checkpoint_path": ws.get("sam2_checkpoint_path", ""),
                "sam3_checkpoint_path": ws.get("sam3_checkpoint_path"),
                "sam3_bpe_path": ws.get("sam3_bpe_path"),
                "keep_models_loaded": ws.get("keep_models_loaded", False),
                "max_side": ws.get("max_side", 1024),
                "images_dir": str(ws["images_dir"]) if ws.get("images_dir") else None,
                "save_dir": str(ws["save_dir"]) if ws.get("save_dir") else None,
            }
        )
