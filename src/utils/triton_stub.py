"""Triton stub for platforms where Triton is not available (e.g. macOS).

Vendor SAM3 pulls triton via model_builder → video/tracking modules → tracker_utils
→ edt. Our app only uses image inference (build_sam3_image_model, predict_mask_*);
we never run video/tracking code that calls edt_triton. So we only need to fake
triton so the import chain succeeds (same idea as decord_stub).
"""

from __future__ import annotations

import sys
import types

from src.logging_config import get_logger

logger = get_logger(__name__)


def install_triton_stub_if_needed() -> None:
    """If triton is not installed, put a stub in sys.modules so vendor imports succeed.

    Call before any import that might load sam3 (e.g. before
    import_module("sam3.model_builder")). Our workflow never calls edt_triton.
    """
    if "triton" in sys.modules:
        return
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass

    language = types.ModuleType("triton.language")
    language.constexpr = type(None)  # for annotations like horizontal: tl.constexpr

    stub = types.ModuleType("triton")
    stub.jit = lambda f: f  # no-op so @triton.jit def ... in vendor edt.py just defines the function
    stub.language = language
    stub.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules["triton"] = stub
    sys.modules["triton.language"] = language
    logger.debug("Triton unavailable: stubbed so SAM3 import chain can load (edt not used in image workflow)")
