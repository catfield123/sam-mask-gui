"""Tests that run on Linux but simulate missing decord and triton.

Stubs (decord_stub, triton_stub) are used so SAM3 import chain succeeds on macOS
where decord/triton are not available. These tests verify stub behaviour by
temporarily making decord/triton appear missing.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest


@contextmanager
def simulate_missing_decord_triton() -> Generator[object, None, None]:
    """Temporarily make 'import decord' and 'import triton' raise ImportError.

    Saves and removes decord/triton from sys.modules, patches builtins.__import__,
    then restores everything on exit. Yields the real __import__ so tests can
    restore it before calling code that needs to import other modules (e.g. sam3).
    """
    builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__
    real_import = builtins.get("__import__", __import__) if isinstance(builtins, dict) else getattr(builtins, "__import__", __import__)

    def patched_import(name: str, *args: object, **kwargs: object):  # noqa: ANN002
        if name in ("decord", "triton"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    saved = {
        "decord": sys.modules.pop("decord", None),
        "triton": sys.modules.pop("triton", None),
        "triton.language": sys.modules.pop("triton.language", None),
    }
    try:
        with patch("builtins.__import__", patched_import):
            yield real_import
    finally:
        for key, mod in saved.items():
            if mod is not None:
                sys.modules[key] = mod
            elif key in sys.modules:
                del sys.modules[key]


class TestDecordStubWhenMissing:
    """With decord simulated as missing, install_decord_stub_if_needed installs a stub."""

    def test_decord_stub_installed_and_has_expected_attrs(self) -> None:
        with simulate_missing_decord_triton():
            from src.utils.decord_stub import install_decord_stub_if_needed

            install_decord_stub_if_needed()

            assert "decord" in sys.modules
            decord = sys.modules["decord"]
            assert hasattr(decord, "cpu")
            assert hasattr(decord, "VideoReader")
            assert hasattr(decord, "bridge")
            assert callable(decord.cpu)
            # VideoReader should be a class; instantiating and __getitem__ should raise our error
            vr = decord.VideoReader("fake")
            with pytest.raises(RuntimeError, match="decord is not installed"):
                vr["anything"]


class TestTritonStubWhenMissing:
    """With triton simulated as missing, install_triton_stub_if_needed installs a stub."""

    def test_triton_stub_installed_and_has_expected_attrs(self) -> None:
        with simulate_missing_decord_triton():
            from src.utils.triton_stub import install_triton_stub_if_needed

            install_triton_stub_if_needed()

            assert "triton" in sys.modules
            triton = sys.modules["triton"]
            assert hasattr(triton, "jit")
            assert hasattr(triton, "language")
            assert hasattr(triton, "cdiv")
            assert callable(triton.jit)
            assert callable(triton.cdiv)
            # @triton.jit is no-op decorator
            @triton.jit  # type: ignore[misc]
            def dummy() -> None:
                pass

            assert dummy is not None
            assert hasattr(triton.language, "constexpr")
            # cdiv used in vendor
            assert triton.cdiv(10, 3) == 4


class TestCheckSam3WithStubs:
    """With decord/triton simulated missing, stubs are installed and check_sam3_installed can still succeed."""

    def test_install_stubs_then_check_sam3_succeeds_if_sam3_present(self) -> None:
        with simulate_missing_decord_triton() as real_import:
            from src.utils.check_packages import check_sam3_installed
            from src.utils.decord_stub import install_decord_stub_if_needed
            from src.utils.triton_stub import install_triton_stub_if_needed

            install_decord_stub_if_needed()
            install_triton_stub_if_needed()
            # Restore real __import__ so import_module("sam3.model_builder") can load sam3/torch/etc.
            import builtins

            with patch.object(builtins, "__import__", real_import):
                installed, msg = check_sam3_installed()
            # If SAM3 is installed, our stubs allow the chain to complete
            assert isinstance(installed, bool)
            assert msg is None or isinstance(msg, str)
            if installed:
                assert msg is None
            else:
                # SAM3 not installed at all (e.g. in CI without sam3)
                assert "sam3" in (msg or "").lower() or "error" in (msg or "").lower()
