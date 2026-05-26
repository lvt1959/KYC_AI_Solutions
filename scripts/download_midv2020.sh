#!/usr/bin/env bash
# Download GenMRP+MIDV-2020 dataset from Roboflow (recommended)
# or fallback to official MIDV-2020 photo split.
#
# Usage:
#   bash scripts/download_midv2020.sh              # Roboflow (needs ROBOFLOW_API_KEY env var)
#   bash scripts/download_midv2020.sh --legacy     # Official MIDV-2020 sFTP (~2 GB)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ──────────────────────────────────────────────────
# Roboflow download (default, recommended)
# ──────────────────────────────────────────────────
download_roboflow() {
    if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
        echo "Error: ROBOFLOW_API_KEY not set."
        echo ""
        echo "Get your free API key:"
        echo "  1. Create account at https://app.roboflow.com (free)"
        echo "  2. Go to https://app.roboflow.com/settings/api"
        echo "  3. Copy your Private API Key"
        echo "  4. export ROBOFLOW_API_KEY=your_key_here"
        exit 1
    fi

    echo "Downloading GenMRP+MIDV-2020 from Roboflow..."
    python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='${ROBOFLOW_API_KEY}')
project = rf.workspace('maastricht-university').project('genmrp-midv-2020')
version = project.version(1)
dataset = version.download('yolov11', location='${PROJECT_ROOT}/data/dataset')
print(f'Dataset downloaded to {dataset.location}')
"
    echo "Done. Dataset at ${PROJECT_ROOT}/data/dataset"
}

# ──────────────────────────────────────────────────
# Legacy: official MIDV-2020 sFTP download
# ──────────────────────────────────────────────────
download_legacy() {
    RAW_DIR="${PROJECT_ROOT}/data/raw"
    TARBALL="${RAW_DIR}/photo.tar"

    mkdir -p "${RAW_DIR}"

    if [[ -d "${RAW_DIR}/photo" ]]; then
        echo "MIDV-2020 photo already extracted at ${RAW_DIR}/photo"
        exit 0
    fi

    if [[ ! -f "${TARBALL}" ]]; then
        echo "Downloading MIDV-2020 photo split (~2 GB)..."
        if ! curl -L --fail -o "${TARBALL}" "ftp://smartengines.com/midv-2020/photo.tar"; then
            echo ""
            echo "FTP download failed. Falling back to manual instructions:"
            echo "    1. Visit https://l3i-share.univ-lr.fr/MIDV2020/"
            echo "    2. Fill the access form, download photo.tar"
            echo "    3. Place the file at ${TARBALL}"
            echo "    4. Re-run this script with --legacy"
            exit 1
        fi
    fi

    echo "Extracting..."
    tar -xf "${TARBALL}" -C "${RAW_DIR}/"
    echo "Done. Dataset at ${RAW_DIR}/photo"
}

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
if [[ "${1:-}" == "--legacy" ]]; then
    download_legacy
else
    download_roboflow
fi
