"""Tests for ConfigService."""

import json
from pathlib import Path

import pytest

from src.services.config_service import ConfigService


@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    """A temporary config file path (file may not exist)."""
    return tmp_path / "config.json"


@pytest.fixture
def config_service(temp_config_path: Path) -> ConfigService:
    """ConfigService pointing at a temp config file."""
    return ConfigService(temp_config_path)


class TestConfigServiceLoad:
    """Tests for ConfigService.load."""

    def test_load_missing_file_returns_empty_dict(self, config_service: ConfigService) -> None:
        """When config file does not exist, load returns {}."""
        assert config_service.load() == {}

    def test_load_valid_json_returns_dict(
        self, config_service: ConfigService, temp_config_path: Path
    ) -> None:
        """When config file has valid JSON, load returns the parsed dict."""
        data = {"max_side": 1024, "sam2_checkpoint_path": "/some/path"}
        temp_config_path.write_text(json.dumps(data), encoding="utf-8")
        assert config_service.load() == data

    def test_load_corrupt_json_returns_empty_and_deletes_file(
        self, config_service: ConfigService, temp_config_path: Path
    ) -> None:
        """When config file is corrupt, load returns {} and removes the file."""
        temp_config_path.write_text("not valid json {", encoding="utf-8")
        result = config_service.load()
        assert result == {}
        assert not temp_config_path.exists()


class TestConfigServiceSave:
    """Tests for ConfigService.save."""

    def test_save_persists_dict(
        self, config_service: ConfigService, temp_config_path: Path
    ) -> None:
        """save writes the config to disk and returns True."""
        data = {"max_side": 512}
        assert config_service.save(data) is True
        assert json.loads(temp_config_path.read_text(encoding="utf-8")) == data


class TestConfigServiceGetters:
    """Tests for ConfigService get_* methods."""

    def test_get_model_type_default(self, config_service: ConfigService) -> None:
        """get_model_type returns 'sam2' when key missing."""
        assert config_service.get_model_type({}) == "sam2"

    def test_get_sam2_checkpoint_path_fallback(self, config_service: ConfigService) -> None:
        """get_sam2_checkpoint_path falls back to checkpoint_path."""
        config = {"checkpoint_path": "/old/path"}
        assert config_service.get_sam2_checkpoint_path(config) == "/old/path"
        config["sam2_checkpoint_path"] = "/new/path"
        assert config_service.get_sam2_checkpoint_path(config) == "/new/path"

    def test_get_max_side_default(self, config_service: ConfigService) -> None:
        """get_max_side returns 1024 when key missing."""
        assert config_service.get_max_side({}) == 1024

    def test_get_sam3_paths_none_when_missing(self, config_service: ConfigService) -> None:
        """get_sam3_checkpoint_path and get_sam3_bpe_path return None when keys missing or empty."""
        assert config_service.get_sam3_checkpoint_path({}) is None
        assert config_service.get_sam3_bpe_path({}) is None
