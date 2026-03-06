"""Service for loading and saving application configuration."""

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


class ConfigService:
    """Reads and writes the JSON configuration file.

    Args:
        - config_path (Path): Filesystem path to the JSON config file.
    """

    def __init__(self, config_path: Path):
        """Initialise the config service.

        Args:
            - config_path (Path): Path to the ``config.json`` file.
        """
        self.config_path = config_path
        logger.debug("ConfigService initialised with path=%s", config_path)

    def load(self) -> Dict[str, Any]:
        """Load configuration from disk.

        Returns:
            - dict: Parsed config dictionary, or empty dict if the file is
              missing or corrupted (corrupted files are deleted automatically).
        """
        logger.debug("Loading config from %s", self.config_path)
        if not self.config_path.exists():
            logger.debug("Config file does not exist, returning empty dict")
            return {}

        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Config loaded successfully from %s (keys: %s)", self.config_path, list(data.keys()))
            return data
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.error("Error loading config: %s. Deleting config file.", e)
            with contextlib.suppress(Exception):
                self.config_path.unlink()
            return {}

    def save(self, config: Dict[str, Any]) -> bool:
        """Persist configuration to disk.

        Args:
            - config (dict): Configuration dictionary to serialise.

        Returns:
            - bool: ``True`` on success, ``False`` on failure.
        """
        logger.debug("Saving config to %s", self.config_path)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("Config saved successfully to %s", self.config_path)
            return True
        except Exception as e:
            logger.error("Error saving config: %s", e)
            return False

    def get_model_type(self, config: Dict[str, Any]) -> str:
        """Extract the model type from a config dict."""
        return config.get("model_type", "sam2")

    def get_checkpoint_path(self, config: Dict[str, Any]) -> str:
        """Extract the SAM2 checkpoint path from a config dict.

        For backward compatibility, also checks "checkpoint_path" key.
        """
        return config.get("sam2_checkpoint_path", config.get("checkpoint_path", ""))

    def get_sam2_checkpoint_path(self, config: Dict[str, Any]) -> str:
        """Extract the SAM2 checkpoint path from a config dict."""
        return config.get("sam2_checkpoint_path", config.get("checkpoint_path", ""))

    def get_sam3_checkpoint_path(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract the SAM3 checkpoint path from a config dict."""
        path = config.get("sam3_checkpoint_path")
        return path if path else None

    def get_sam3_bpe_path(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract the SAM3 BPE tokenizer path from a config dict."""
        path = config.get("sam3_bpe_path")
        return path if path else None

    def get_keep_models_loaded(self, config: Dict[str, Any]) -> bool:
        """Return whether the UI should try to keep both models in memory."""
        return bool(config.get("keep_models_loaded", False))

    def get_max_side(self, config: Dict[str, Any]) -> int:
        """Extract the max-side scaling value from a config dict."""
        return config.get("max_side", 1024)

    def get_images_dir(self, config: Dict[str, Any]) -> Optional[Path]:
        """Extract the images directory from a config dict.

        Returns:
            - Path | None: The directory path if it exists, otherwise ``None``.
        """
        images_dir = config.get("images_dir")
        if images_dir and Path(images_dir).exists() and Path(images_dir).is_dir():
            return Path(images_dir)
        return None

    def get_save_dir(self, config: Dict[str, Any]) -> Optional[Path]:
        """Extract the mask-save directory from a config dict.

        Returns:
            - Path | None: The directory path if it exists, otherwise ``None``.
        """
        save_dir = config.get("save_dir")
        if save_dir and Path(save_dir).exists() and Path(save_dir).is_dir():
            return Path(save_dir)
        return None
