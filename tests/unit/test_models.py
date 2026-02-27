"""Tests for data models (Keypoint, ImageState, etc.)."""

from pathlib import Path

import pytest

from src.models import ImageState, Keypoint, KeypointType
from src.models.session_models import BatchSession, FrameBackup


class TestKeypoint:
    """Tests for Keypoint and KeypointType."""

    def test_keypoint_creation(self) -> None:
        """Keypoint stores x, y and type."""
        kp = Keypoint(x=10, y=20, type=KeypointType.POSITIVE)
        assert kp.x == 10
        assert kp.y == 20
        assert kp.type == KeypointType.POSITIVE

    def test_keypoint_type_negative(self) -> None:
        """KeypointType.NEGATIVE can be used."""
        kp = Keypoint(x=0, y=0, type=KeypointType.NEGATIVE)
        assert kp.type == KeypointType.NEGATIVE


class TestImageState:
    """Tests for ImageState."""

    def test_image_state_creation(self, tmp_path: Path) -> None:
        """ImageState has path and default keypoints/mask."""
        path = tmp_path / "image.png"
        path.touch()
        state = ImageState(path=path)
        assert state.path == path
        assert state.keypoints == []
        assert state.mask is None

    def test_image_state_with_keypoints(self, tmp_path: Path) -> None:
        """ImageState can hold keypoints."""
        path = tmp_path / "image.png"
        path.touch()
        kp = Keypoint(x=5, y=5, type=KeypointType.POSITIVE)
        state = ImageState(path=path, keypoints=[kp])
        assert len(state.keypoints) == 1
        assert state.keypoints[0].x == 5


class TestFrameBackup:
    """Tests for FrameBackup."""

    def test_frame_backup_creation(self, tmp_path: Path) -> None:
        """FrameBackup stores image_path, frame_idx, and optional masks."""
        path = tmp_path / "img.png"
        path.touch()
        backup = FrameBackup(image_path=path, frame_idx=0)
        assert backup.image_path == path
        assert backup.frame_idx == 0
        assert backup.old_mask is None
        assert backup.new_mask is None


class TestBatchSession:
    """Tests for BatchSession."""

    def test_batch_session_creation(self) -> None:
        """BatchSession stores operation_type and frame_backups."""
        session = BatchSession(operation_type="video_propagation", frame_backups=[])
        assert session.operation_type == "video_propagation"
        assert session.frame_backups == []
        assert session.saved_to_disk is False
        assert session.undone is False
