"""Install a decord stub into sys.modules when decord is not available (e.g. macOS).

Vendor SAM3 pulls decord via model_builder → sam3_image → collator → sam3_image_dataset.
Our app only uses image inference; video loading is never called. So we can stub decord
so that the import chain succeeds. If vendor code ever tried to load video, it would
raise a clear error.
"""

import sys
import types


def install_decord_stub_if_needed() -> None:
    """If decord is not installed, put a stub module in sys.modules['decord'].

    Call this before any import that might load sam3 (e.g. before
    import_module("sam3.model_builder")). Then vendor's "from decord import cpu,
    VideoReader" will get our stub and the import chain succeeds.
    """
    if "decord" in sys.modules:
        return
    try:
        import decord  # noqa: F401  # type: ignore[import-untyped]
        return
    except ImportError:
        pass

    class _VideoReaderStub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __getitem__(self, key: object) -> object:
            raise RuntimeError(
                "decord is not installed. Video loading is not supported on this "
                "platform (e.g. macOS). Install decord via conda or build from source "
                "if you need video."
            )

        def __iter__(self) -> object:
            raise RuntimeError(
                "decord is not installed. Video loading is not supported on this platform."
            )

        def asnumpy(self) -> object:
            raise RuntimeError("decord is not installed. Video loading is not supported.")

    def _cpu_stub(*args: object, **kwargs: object) -> object:
        return object()

    stub = types.ModuleType("decord")
    stub.cpu = _cpu_stub
    stub.VideoReader = _VideoReaderStub
    stub.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
    sys.modules["decord"] = stub
