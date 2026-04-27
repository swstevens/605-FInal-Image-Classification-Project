# AI Image Detection — 605 Final Project

Trains **EfficientNet-B0** (CNN) and **DeiT-Tiny** (Transformer) on the [ArtiFact](https://www.kaggle.com/datasets/awsaf49/artifact-dataset) dataset to detect AI-generated images across multiple generators.

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create the environment and install dependencies

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies automatically.

> **CUDA users:** The default `torch` from PyPI includes CUDA support on Linux and Windows. If you need a specific CUDA version (e.g. cu118, cu121), install torch separately first:
> ```bash
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> uv sync --no-install-package torch --no-install-package torchvision
> ```

### 3. Set up Kaggle credentials

Download your API token from [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New API Token** and place it at:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

On Linux/macOS, restrict permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Launch Jupyter

```bash
uv run jupyter lab
```

Then open `train_ai_detection_local.ipynb` and run cells top-to-bottom.

## Notebooks

| Notebook | Description |
|---|---|
| `train_ai_detection_local.ipynb` | Local training — checkpoints saved to `./checkpoints/` |
| `train_ai_detection.ipynb` | Google Colab version — checkpoints saved to Google Drive |

## Notes

- The full ArtiFact dataset is ~70 GB. Set `MAX_IMAGES_PER_GENERATOR = 5000` (default) to do a fast end-to-end test before a full run.
- Training on CPU is very slow; a CUDA GPU is strongly recommended.
- ONNX exports land alongside the `.pt` checkpoints and are compatible with ONNX Runtime Web.
