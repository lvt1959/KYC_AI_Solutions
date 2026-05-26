#!/usr/bin/env bash
# Download MIDV-2020 photo split (~2 GB) into data/raw/
# Mirrors:
#   - ftp://smartengines.com/midv-2020/photo.tar
#   - https://l3i-share.univ-lr.fr/MIDV2020/ (manual form)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TARBALL="${RAW_DIR}/photo.tar"

mkdir -p "${RAW_DIR}"

if [[ -d "${RAW_DIR}/photo" ]]; then
    echo "✅ MIDV-2020 photo already extracted at ${RAW_DIR}/photo"
    exit 0
fi

if [[ ! -f "${TARBALL}" ]]; then
    echo "⏬ Downloading MIDV-2020 photo split (~2 GB)…"
    if ! curl -L --fail -o "${TARBALL}" "ftp://smartengines.com/midv-2020/photo.tar"; then
        echo ""
        echo "⚠️  FTP download failed. Falling back to manual instructions:"
        echo "    1. Visit https://l3i-share.univ-lr.fr/MIDV2020/"
        echo "    2. Fill the access form, download photo.tar"
        echo "    3. Place the file at ${TARBALL}"
        echo "    4. Re-run this script"
        exit 1
    fi
fi

echo "📂 Extracting…"
tar -xf "${TARBALL}" -C "${RAW_DIR}/"

echo "✅ Done. Dataset at ${RAW_DIR}/photo"
