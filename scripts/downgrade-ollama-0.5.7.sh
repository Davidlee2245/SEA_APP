#!/usr/bin/env bash
# Downgrade Ollama to v0.5.7 (RTX 5090 / CUDA 12 + mllama / llama3.2-vision).
# Usage: bash scripts/downgrade-ollama-0.5.7.sh

set -euo pipefail

VERSION="v0.5.7"
ARCHIVE_URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-amd64.tgz"
ARCHIVE="/tmp/ollama-${VERSION}.tgz"
INSTALL_ROOT="/usr/share/ollama-${VERSION}"
DROP_IN="/etc/systemd/system/ollama.service.d/override.conf"
SEA_CONFIG="/home/mhl/.config/sea-exosome-analysis/sea-config.json"
MODEL_TAG="llama3.2-vision:11b"

echo "=== Step 1: Stop ollama ==="
sudo systemctl stop ollama
echo "OK"

echo ""
echo "=== Step 2: Download ${VERSION} ==="
curl -L "${ARCHIVE_URL}" -o "${ARCHIVE}"
ls -lh "${ARCHIVE}"
echo "OK"

echo ""
echo "=== Step 3: Install to ${INSTALL_ROOT} ==="
sudo rm -rf /usr/share/ollama-v*
sudo mkdir -p "${INSTALL_ROOT}"
sudo tar -xzf "${ARCHIVE}" -C "${INSTALL_ROOT}"
sudo cp "${INSTALL_ROOT}/bin/ollama" /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama
echo "OK"

echo ""
echo "=== Step 4: systemd override ==="
sudo mkdir -p "$(dirname "${DROP_IN}")"
sudo tee "${DROP_IN}" >/dev/null <<EOF
[Service]
User=mhl
Group=mhl
Environment="HOME=/home/mhl"
Environment="OLLAMA_MODELS=/home/mhl/.ollama/models"
Environment="LD_LIBRARY_PATH=${INSTALL_ROOT}/lib/ollama"
EOF
cat "${DROP_IN}"
echo "OK"

echo ""
echo "=== Step 5: Start ollama ==="
sudo systemctl daemon-reload
sudo systemctl start ollama
sleep 3
echo "OK"

echo ""
echo "=== Step 6: Verify version ==="
ollama --version
curl -s http://127.0.0.1:11434/api/version
echo ""

echo ""
echo "=== Step 7: Test ${MODEL_TAG} ==="
ollama run "${MODEL_TAG}" "hi"
echo "OK"

echo ""
echo "=== Step 8: Update sea-config.json ==="
mkdir -p "$(dirname "${SEA_CONFIG}")"
python3 - <<PY
import json
from pathlib import Path
p = Path("${SEA_CONFIG}")
data = json.loads(p.read_text()) if p.exists() else {}
data["llmProvider"] = data.get("llmProvider") or "ollama"
data["ollamaModel"] = "${MODEL_TAG}"
p.write_text(json.dumps(data, indent=2) + "\n")
print(p.read_text())
PY

echo ""
echo "=== Done ==="
ollama list
