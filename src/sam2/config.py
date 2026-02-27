"""SAM2 checkpoint-to-config mapping and supported image extensions."""

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def cfg_for_ckpt(ckpt_path: str) -> str:
    """Return the YAML config path corresponding to a SAM2 checkpoint file.

    Args:
        - ckpt_path (str): Filesystem path to the SAM2 ``.pt`` checkpoint.

    Returns:
        - str: Relative path to the matching YAML config file.

    Raises:
        - ValueError: If the checkpoint filename is not recognised.
    """
    name = Path(ckpt_path).name

    # SAM 2.1
    if name == "sam2.1_hiera_tiny.pt":
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    if name == "sam2.1_hiera_small.pt":
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if name == "sam2.1_hiera_base_plus.pt":
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if name == "sam2.1_hiera_large.pt":
        return "configs/sam2.1/sam2.1_hiera_l.yaml"

    # SAM 2
    if name == "sam2_hiera_tiny.pt":
        return "configs/sam2/sam2_hiera_t.yaml"
    if name == "sam2_hiera_small.pt":
        return "configs/sam2/sam2_hiera_s.yaml"
    if name == "sam2_hiera_base_plus.pt":
        return "configs/sam2/sam2_hiera_b+.yaml"
    if name == "sam2_hiera_large.pt":
        return "configs/sam2/sam2_hiera_l.yaml"

    raise ValueError(f"Unknown checkpoint name: {name}")
