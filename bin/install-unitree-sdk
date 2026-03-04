#!/usr/bin/env bash
# Install Unitree SDK and prerequisites into the dimos .venv (uses uv)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"
SDK2_PATH="${SDK2_PATH:-/opt/unitree_sdk2_python}"

# Fallback to user-local path if /opt is not writable and we don't have sudo
if [[ ! -d "$SDK2_PATH" ]] && ! sudo -n true 2>/dev/null; then
    SDK2_PATH="$HOME/unitree_sdk2_python"
fi

echo "=== Unitree SDK Install ==="
echo "  repo:            $REPO_DIR"
echo "  venv python:     $VENV_PYTHON"
echo "  CYCLONEDDS_HOME: $CYCLONEDDS_HOME"
echo "  SDK2_PATH:       $SDK2_PATH"
echo ""

# ── 1. Ensure venv ──────────────────────────────────────────────────────────
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "--- Creating venv with uv sync ---"
    (cd "$REPO_DIR" && uv sync)
fi

# ── 2. Ensure cmake is available ────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    echo "--- Installing cmake via nix ---"
    nix profile install nixpkgs#cmake
fi

# ── 3. Build CycloneDDS from source ─────────────────────────────────────────
if [[ ! -d "$CYCLONEDDS_HOME" ]]; then
    echo "--- Building CycloneDDS from source ---"
    if [[ ! -d "$HOME/cyclonedds" ]]; then
        git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x ~/cyclonedds
    fi
    mkdir -p ~/cyclonedds/build ~/cyclonedds/install
    (
        cd ~/cyclonedds/build
        cmake .. -DCMAKE_INSTALL_PREFIX=../install
        cmake --build . --target install -j"$(nproc)"
    )
fi

# ── 4. Clone unitree_sdk2_python ─────────────────────────────────────────────
if [[ ! -d "$SDK2_PATH" ]]; then
    echo "--- Cloning unitree_sdk2_python to $SDK2_PATH ---"
    if sudo -n true 2>/dev/null; then
        sudo git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK2_PATH"
        sudo chown -R "$USER":"$USER" "$SDK2_PATH"
    else
        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK2_PATH"
    fi
fi

# ── 5. Install CycloneDDS Python bindings ───────────────────────────────────
echo "--- Installing cyclonedds==0.10.2 ---"
CYCLONEDDS_HOME="$CYCLONEDDS_HOME" uv pip install --python "$VENV_PYTHON" "cyclonedds==0.10.2"

# ── 6. Install unitree_sdk2py (editable) ────────────────────────────────────
echo ""
echo "--- Installing unitree_sdk2py from $SDK2_PATH ---"
CYCLONEDDS_HOME="$CYCLONEDDS_HOME" uv pip install --python "$VENV_PYTHON" --no-deps -e "$SDK2_PATH"
uv pip install --python "$VENV_PYTHON" "numpy<2.0,>=1.26" "opencv-python"

# ── 7. Install unitree-webrtc-connect-leshy ──────────────────────────────────
echo ""
echo "--- Installing unitree-webrtc-connect-leshy ---"
uv pip install --python "$VENV_PYTHON" "unitree-webrtc-connect-leshy>=2.0.7"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To validate:"
echo "  $VENV_PYTHON -c \"import unitree_sdk2py; print('unitree_sdk2py OK')\""
echo "  $VENV_PYTHON -c \"from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection; print('webrtc OK')\""
