# SAM2/SAM3 Mask Creation GUI

GUI application for creating masks from images using [SAM2](https://github.com/facebookresearch/sam2) and [SAM3](https://github.com/facebookresearch/sam3).

## Features

- **Point-based segmentation (SAM2):** place positive (left click) and negative (right click) points. Masks update in real time.
- **Brush refinement:** hold **Shift** for brush mode; paint to add or subtract from the mask. **Shift + scroll** changes brush size. **Ctrl + scroll** zooms; **middle mouse drag** pans.
- **Alt:** hold **Alt** for mask preview (see current mask without overlay).
- **Text-prompt segmentation (SAM3):** describe the object in text to segment the current image or a batch of selected images.
- **Propagate masks (SAM2):** propagate masks from key-frames to all images in order (“Propagate Masks” button).
- **Grow / shrink mask:** expand or contract the mask boundary by a number of pixels (single image or selected images).
- **Batch operations:** select multiple images (Ctrl+click), run “Grow Mask For Selected” or “Segment Selected by Prompt”; then “Save All” or “Revert”.
- **Undo / Redo** for point and brush changes (per image).
- **Auto-load existing masks** when opening a folder; optional scaling (max_side) to save memory; masks are saved at original resolution.
- **Settings:** SAM2/SAM3 checkpoint paths, max_side, mask colour, opacity, keep-both-models loaded option.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — Python package manager (recommended)
- CUDA (optional, for GPU)
- SAM2 and/or SAM3 checkpoints (see below)

## Installation

### 1. Clone the repository

```bash
git clone <this-repo-url>
cd auto_segmentation
```

To use bundled SAM2/SAM3 via submodules:

```bash
git submodule update --init --recursive
```

### 2. Create virtual environment and install dependencies

Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows

uv pip install -e .
```

This installs the project in editable mode and all dependencies from `pyproject.toml` (PyQt6, torch, numpy, SAM3-related libs like einops/decord, and setuptools &lt;82 for SAM3 compatibility).

### 3. Install SAM2 and/or SAM3 (optional)

The app works with the external `sam2` and `sam3` packages. You can install them from the repo’s submodules (if you ran `git submodule update --init --recursive`):

```bash
uv pip install -e vendor/sam2
# Optional, for text prompts and SAM3 features:
uv pip install -e vendor/sam3
```

Alternatively, clone SAM2/SAM3 elsewhere and install in editable mode:

```bash
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2
uv pip install -e /path/to/sam2
```

### 4. Download checkpoints

- **SAM2**: e.g. [sam2.1_hiera_small.pt](https://github.com/facebookresearch/sam2) (balance of speed and quality). Place the file and set its path in Settings.
- **SAM3**: Weights are hosted on [Hugging Face (facebook/sam3)](https://huggingface.co/facebook/sam3). The model is **gated**: you must open the model page, request access, and wait until your request is approved before you can download the checkpoint. Then set the checkpoint path in Settings (and optionally the BPE path; see below).

## Run

In activated venv run:

```bash
python -m src.main
```

**First run:** A settings dialog will open. Set at least one checkpoint path (SAM2 or SAM3) and, if you like, **max_side** (see [Memory](#memory) below). Then open an images folder and set a save folder.

**Debug logging:**

```bash
python -m src.main --debug
```

## Usage

- **Points:** Green = positive, red = negative. Red overlay = current mask. Checkmark in the list = mask saved.
- **Help → Keyboard shortcuts** in the app shows all shortcuts in a window.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+O** | Open images folder |
| **Ctrl+S** | Save current mask |
| **Ctrl+Z** | Undo |
| **Ctrl+Y** | Redo |
| **G** | Grow / shrink current mask (apply value from right panel) |
| **←** (Left) | Previous image |
| **→** (Right) | Next image |
| **Shift** | Brush mode (hold while painting) |
| **Alt** | Mask preview (hold to see mask without overlay) |
| **Ctrl + scroll** | Zoom in/out |
| **Shift + scroll** | Change brush size |
| **Middle mouse drag** | Pan the image |

Other actions (Set Save Folder, Settings, Exit, Clear mask, Segment by prompt, etc.) use the menu or buttons.

## Project structure

```
auto_segmentation/
├── src/
│   ├── main.py           # Entry point
│   ├── models/           # Data models (keypoints, image state)
│   ├── sam2/             # SAM2 integration (wrapper)
│   ├── sam3/             # SAM3 integration (wrapper)
│   ├── gui/              # Panels, controllers, dialogs, workers
│   ├── services/         # Config, image, mask services
│   └── utils/            # Helpers (e.g. package checks)
├── vendor/               # Optional: sam2, sam3 submodules
├── config.json           # App config (created/updated by app)
├── pyproject.toml
└── README.md
```

## Development

From the project root with the venv activated:

```bash
ruff check src/ && ruff format src/   # Lint and format
mypy -p src                            # Type check
pytest tests/ -v                       # Run tests
```

## Memory

If you run out of GPU or RAM (e.g. large images or multiple models loaded), use **Settings** and reduce **Max side size** (max_side). This limit applies only while working in the app: images are scaled down for display and for running the models, so less memory is used. **When you save a mask, it is always written at the original image resolution** (upscaled if needed). So you can safely use e.g. **512** or **768** to reduce memory; masks on disk will still be full resolution. Use **0** for no limit (original size in memory, highest use).

## BPE path (SAM3)

**BPE** (Byte Pair Encoding) is the tokenizer that turns your text prompts into tokens for SAM3’s text encoder. You **do not need** to set the BPE path: the SAM3 package ships with a built-in vocabulary (`bpe_simple_vocab_16e6.txt.gz`), and the app uses it when the field is left empty. Only set a custom BPE path in Settings if you have a specific tokenizer file you want to use.

## Troubleshooting

- **CUDA not available:** The app will run on CPU, but it will be slower. If you expected GPU acceleration, check that CUDA and the matching PyTorch build are installed (e.g. `python -c "import torch; print(torch.cuda.is_available())"`). If that prints `False`, install a CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org). Either way, the app will still run on CPU if no GPU is detected.
- **Checkpoint load error:** Ensure the path in Settings is correct, the file exists, and the matching SAM package is installed (`uv pip install -e vendor/sam2` or your SAM path).
- **Out of memory:** Lower **max_side** in Settings (see [Memory](#memory) above).

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file in the repository root.

**Third-party code in this repository:**

- **SAM2** (Segment Anything 2) is licensed under the **Apache License 2.0**. Code in `vendor/sam2/` is distributed under that license; see [vendor/sam2/LICENSE](vendor/sam2/LICENSE).
- **SAM3** (Segment Anything 3) is licensed under the **SAM License** (Meta). Code in `vendor/sam3/` is distributed under that license; see [vendor/sam3/LICENSE](vendor/sam3/LICENSE).

Use of SAM2 and SAM3 is subject to their respective license terms (including any use restrictions in the SAM License, such as prohibited military or nuclear applications).
