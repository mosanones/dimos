#!/bin/bash
# Build script for GraspGen Docker image
#
# This script:
# 1. Ensures GraspGen model checkpoints are downloaded via Git LFS
# 2. Builds the Docker image from the repo root context
#
# Usage:
#   ./build.sh              # Build with default tag
#   ./build.sh my-tag       # Build with custom tag

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
IMAGE_TAG="${1:-dimos-graspgen}"

echo "=========================================="
echo "GraspGen Docker Build"
echo "=========================================="
echo "Repo root:  ${REPO_ROOT}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Image tag:  ${IMAGE_TAG}"
echo ""

# Step 1: Ensure GraspGen data is available
echo "[1/2] Checking GraspGen checkpoints..."
CHECKPOINTS_DIR="${REPO_ROOT}/data/graspgen/checkpoints"

if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "  Checkpoints not found. Downloading via get_data()..."
    cd "${REPO_ROOT}"
    python3 -c "from dimos.utils.data import get_data; get_data('graspgen')"
fi

# Verify checkpoints exist
if [ ! -f "${CHECKPOINTS_DIR}/graspgen_robotiq_2f_140_gen.pth" ]; then
    echo "ERROR: Checkpoint files not found in ${CHECKPOINTS_DIR}"
    echo ""
    echo "To download checkpoints, run:"
    echo "  python -c \"from dimos.utils.data import get_data; get_data('graspgen')\""
    exit 1
fi

echo "  Checkpoints found at ${CHECKPOINTS_DIR}"
ls -lh "${CHECKPOINTS_DIR}"/*.pth | head -3
echo ""

# Step 2: Build Docker image
echo "[2/2] Building Docker image..."
cd "${REPO_ROOT}"
docker build -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Image: ${IMAGE_TAG}"
echo ""
echo "To run:"
echo "  docker run --gpus all -p 8094:8094 --name dimos_graspgen ${IMAGE_TAG}"
