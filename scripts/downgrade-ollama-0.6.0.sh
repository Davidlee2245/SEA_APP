#!/usr/bin/env bash
# Downgrade Ollama to v0.6.0 (mllama / llama3.2-vision stability).
# NOTE: v0.6.0 ships as ollama-linux-amd64.tgz (not a single ollama-linux-amd64 binary).
# Usage: bash scripts/downgrade-ollama-0.6.0.sh

set -euo pipefail

VERSION="v0.6.0"
TGZ_URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-amd64.tgz"
INSTALL_ROOT="/usr/share/ollama-${VERSION}"
SEA_CONFIG="/home/mhl/.config/sea-exosome-analysis/sea-config.json"
MODEL_TAG="llama3.2-vision:11b"
USER_LOCAL="/home/mhl/.local/ollama-0.6.0"
TMP_DIR="/tmp/ollama-downgrade-${VERSION}"

echo "=== Step 1: Stop ollama ==="
sudo systemctl stop ollama
echo "OK"

echo ""
echo "=== Step 2: Download and install ${VERSION} (system + user) ==="
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
curl -L "${TGZ_URL}" -o "${TMP_DIR}/ollama-linux-amd64.tgz"
echo "Extracting (this includes bin/ollama and lib/ollama/*.so)..."
tar -xzf "${TMP_DIR}/ollama-linux-amd64.tgz" -C "${TMP_DIR}"

sudo rm -rf /usr/lib/ollama "${INSTALL_ROOT}"
sudo mkdir -p "${INSTALL_ROOT}"
sudo cp -r "${TMP_DIR}/bin" "${TMP_DIR}/lib" "${INSTALL_ROOT}/"
sudo cp "${INSTALL_ROOT}/bin/ollama" /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama

mkdir -p "${USER_LOCAL}"
rm -rf "${USER_LOCAL:?}"/*
cp -r "${TMP_DIR}/bin" "${TMP_DIR}/lib" "${USER_LOCAL}/"
cp "${USER_LOCAL}/bin/ollama" "${HOME}/.local/bin/ollama"
chmod +x "${HOME}/.local/bin/ollama"
echo "OK: installed to ${INSTALL_ROOT} and ${USER_LOCAL}"

echo ""
echo "=== Step 3: systemd override (libs + models path) ==="
DROP_IN="/etc/systemd/system/ollama.service.d/override.conf"
sudo mkdir -p "$(dirname "${DROP_IN}")"
if [[ -f "${DROP_IN}" ]]; then
  sudo cp "${DROP_IN}" "${DROP_IN}.bak.$(date +%s)"
fi
sudo tee "${DROP_IN}" >/dev/null <<EOF
[Service]
User=mhl
Group=mhl
Environment="HOME=/home/mhl"
Environment="OLLAMA_MODELS=/home/mhl/.ollama/models"
Environment="LD_LIBRARY_PATH=${INSTALL_ROOT}/lib/ollama"
EOF
echo "OK: ${DROP_IN}"

echo ""
echo "=== Step 4: Verify version (client binary; server after start) ==="
export LD_LIBRARY_PATH="${INSTALL_ROOT}/lib/ollama"
/usr/local/bin/ollama --version || true
"${USER_LOCAL}/bin/ollama" --version || true

echo ""
echo "=== Step 5: Start service ==="
sudo systemctl daemon-reload
sudo systemctl start ollama
sleep 3
curl -s http://127.0.0.1:11434/api/version || true
echo ""

echo ""
echo "=== Step 6: Test ${MODEL_TAG} ==="
ollama run "${MODEL_TAG}" "hi"
echo "OK: test run"

echo ""
echo "=== Step 7: Update sea-config.json ==="
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
