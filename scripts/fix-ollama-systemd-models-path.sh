#!/usr/bin/env bash
# Point the Ollama systemd service at existing models in /home/mhl/.ollama/models
# Run: bash scripts/fix-ollama-systemd-models-path.sh   (will prompt for sudo)

set -euo pipefail

MODELS_DIR="/home/mhl/.ollama/models"
DROP_IN_DIR="/etc/systemd/system/ollama.service.d"
DROP_IN_FILE="${DROP_IN_DIR}/override.conf"

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "ERROR: Models directory not found: ${MODELS_DIR}"
  exit 1
fi

echo "==> Creating systemd drop-in: ${DROP_IN_FILE}"
sudo mkdir -p "${DROP_IN_DIR}"
sudo tee "${DROP_IN_FILE}" >/dev/null <<EOF
[Service]
Environment="OLLAMA_MODELS=${MODELS_DIR}"
EOF

echo "==> Allowing User=ollama to read models under ${MODELS_DIR}"
# /home/mhl is typically 750 (drwxr-x---); the ollama user cannot traverse it otherwise.
if command -v setfacl >/dev/null 2>&1; then
  sudo setfacl -m "u:ollama:--x" /home/mhl
  sudo setfacl -m "u:ollama:--x" /home/mhl/.ollama
  sudo setfacl -R -m "u:ollama:rX" "${MODELS_DIR}"
  echo "    Applied ACLs for user ollama"
else
  echo "    setfacl not found — using chmod o+x on /home/mhl and /home/mhl/.ollama"
  sudo chmod o+x /home/mhl /home/mhl/.ollama
  sudo chmod -R o+rX "${MODELS_DIR}"
fi

echo "==> Reloading and restarting ollama"
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 2

echo ""
echo "==> Environment (from systemctl show):"
sudo systemctl show ollama -p Environment --value | tr ' ' '\n' | grep -E 'OLLAMA|HOME' || true

echo ""
echo "==> Service status:"
systemctl is-active ollama && systemctl status ollama --no-pager -l | head -12

echo ""
echo "==> Models visible to API:"
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:11434/api/tags

echo ""
echo "==> Quick run test (should NOT start a multi-GB download if models are found):"
echo "    ollama run llama3.2-vision:90b \"hi\""
echo "Run the command above manually and watch for 'downloading' in journalctl -f"
