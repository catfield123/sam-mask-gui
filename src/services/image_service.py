"""Service for discovering images and resolving mask paths."""

from pathlib import Path
from typing import List

from src.logging_config import get_logger

try:
    from src.sam2.config import IMG_EXTS
except Exception:
    # Keep image discovery working even when the optional SAM2 package cannot be imported.
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

logger = get_logger(__name__)


class ImageService:
    """Stateless helpers for image file discovery and mask path resolution."""

    @staticmethod
    def find_images(directory: Path) -> List[Path]:
        """Find all supported image files in a directory.

        Args:
            - directory (Path): Directory to scan.

        Returns:
            - list[Path]: Sorted list of image file paths.
        """
        logger.debug("find_images: scanning %s", directory)
        if not directory.exists() or not directory.is_dir():
            logger.debug("find_images: directory does not exist or is not a dir")
            return []

        paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMG_EXTS and p.is_file()]
        logger.info("find_images: found %s images in %s", len(paths), directory)
        return paths

    @staticmethod
    def get_mask_path(image_path: Path, save_dir: Path) -> Path:
        """Compute the mask file path for a given image.

        Args:
            - image_path (Path): Source image path.
            - save_dir (Path): Directory where masks are stored.

        Returns:
            - Path: Expected mask file path (``<save_dir>/<stem>.png``).
        """
        return save_dir / f"{image_path.stem}.png"

    @staticmethod
    def mask_exists(image_path: Path, save_dir: Path) -> bool:
        """Check whether a saved mask exists for the given image.

        Args:
            - image_path (Path): Source image path.
            - save_dir (Path): Directory where masks are stored.

        Returns:
            - bool: ``True`` if the mask file exists on disk.
        """
        mask_path = ImageService.get_mask_path(image_path, save_dir)
        return mask_path.exists()
