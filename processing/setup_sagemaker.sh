#!/usr/bin/env bash
# Bootstrap this repository after starting a SageMaker Code Editor application.
#
# Source this file so its PATH update remains active in the current terminal:
#
#     source processing/setup_sagemaker.sh
#
# The installations live beneath the repository/EBS volume and can be reused
# after the Code Editor compute application restarts.

set -euo pipefail

SETUP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETUP_REPOSITORY_DIR="$(cd -- "$SETUP_SCRIPT_DIR/.." && pwd)"
SETUP_TOOLS_PREFIX="$SETUP_REPOSITORY_DIR/.ffmpeg-env"

cd "$SETUP_REPOSITORY_DIR"

echo "Repository: $SETUP_REPOSITORY_DIR"
echo "Persistent tool environment: $SETUP_TOOLS_PREFIX"
df -h "$SETUP_REPOSITORY_DIR"

if command -v conda >/dev/null 2>&1; then
    SETUP_CONDA="$(command -v conda)"
elif [[ -x /opt/conda/bin/conda ]]; then
    SETUP_CONDA=/opt/conda/bin/conda
else
    echo "ERROR: Conda was not found. Use a SageMaker Distribution image that includes Conda." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$SETUP_TOOLS_PREFIX/bin/ffmpeg" \
      || ! -x "$SETUP_TOOLS_PREFIX/bin/ffprobe" \
      || ! -x "$SETUP_TOOLS_PREFIX/bin/tmux" ]]; then
    echo "Installing FFmpeg, FFprobe, and tmux with conda-forge..."
    if [[ -d "$SETUP_TOOLS_PREFIX/conda-meta" ]]; then
        "$SETUP_CONDA" install \
            --yes \
            --prefix "$SETUP_TOOLS_PREFIX" \
            --channel conda-forge \
            ffmpeg \
            tmux
    else
        "$SETUP_CONDA" create \
            --yes \
            --prefix "$SETUP_TOOLS_PREFIX" \
            --channel conda-forge \
            ffmpeg \
            tmux
    fi
else
    echo "Reusing the existing repository-local FFmpeg/tmux environment."
fi

export PATH="$SETUP_TOOLS_PREFIX/bin:$PATH"
hash -r

if ! command -v uv >/dev/null 2>&1; then
    if [[ -x /home/sagemaker-user/.local/bin/uv ]]; then
        export PATH="/home/sagemaker-user/.local/bin:$PATH"
    else
        echo "Installing uv with its official installer..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/home/sagemaker-user/.local/bin:$PATH"
        hash -r
    fi
fi

echo "Synchronizing locked Python dependencies..."
uv sync --project processing --frozen

echo
echo "Installed tool versions:"
ffmpeg -version | sed -n '1p'
ffprobe -version | sed -n '1p'
tmux -V
uv --version

echo
echo "Python/CUDA verification:"
uv run --project processing python -c \
    'import shutil, torch; print("ffmpeg:", shutil.which("ffmpeg")); print("ffprobe:", shutil.which("ffprobe")); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

echo
echo "Setup complete. PATH is active in this terminal."
echo "Start a persistent terminal with: tmux new -s full-g5-run"
echo "Then run:"
echo "  PATH=\"\$PWD/.ffmpeg-env/bin:\$PATH\" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --project processing python processing/00_run_pipeline.py --config processing/runs/full_run_g5_gpu.yaml --ffmpeg \"\$PWD/.ffmpeg-env/bin/ffmpeg\" --ffprobe \"\$PWD/.ffmpeg-env/bin/ffprobe\""
