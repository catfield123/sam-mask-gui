"""Tests for ImageService."""

from pathlib import Path

import pytest

from src.services.image_service import ImageService


class TestFindImages:
    """Tests for ImageService.find_images."""

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """find_images in an empty directory returns []."""
        assert ImageService.find_images(tmp_path) == []

    def test_finds_image_extensions(self, tmp_path: Path) -> None:
        """find_images returns only supported image files, sorted."""
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("no")
        (tmp_path / "d.JPEG").write_bytes(b"x")
        result = ImageService.find_images(tmp_path)
        names = [p.name for p in result]
        assert "a.png" in names
        assert "b.jpg" in names
        assert "d.JPEG" in names
        assert "c.txt" not in names
        assert result == sorted(result, key=lambda p: p.name)

    def test_nonexistent_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """find_images for a path that does not exist returns []."""
        assert ImageService.find_images(tmp_path / "nonexistent") == []


class TestGetMaskPath:
    """Tests for ImageService.get_mask_path."""

    def test_returns_save_dir_with_image_stem_and_png_suffix(self) -> None:
        """get_mask_path returns save_dir / (image_path.stem).png."""
        image_path = Path("/some/folder/image_name.jpg")
        save_dir = Path("/output")
        result = ImageService.get_mask_path(image_path, save_dir)
        assert result == Path("/output/image_name.png")
