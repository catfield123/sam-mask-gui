"""Service for discovering images and resolving mask paths."""

from pathlib import Path
from typing import List

try:
    from src.sam2.config import IMG_EXTS
except Exception:
    # Keep image discovery working even when the optional SAM2 package cannot be imported.
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
        if not directory.exists() or not directory.is_dir():
            return []

        return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMG_EXTS and p.is_file()]

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
