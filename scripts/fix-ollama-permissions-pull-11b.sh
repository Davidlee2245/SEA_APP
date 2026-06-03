#!/usr/bin/env bash
# Fix ~/.ollama permissions, run Ollama as user mhl, pull llama3.2-vision:11b, update SEA config.
# Usage: bash scripts/fix-ollama-permissions-pull-11b.sh

set -euo pipefail

OLLAMA_HOME="/home/mhl/.ollama"
MODELS_DIR="${OLLAMA_HOME}/models"
SEA_CONFIG="/home/mhl/.config/sea-exosome-analysis/sea-config.json"
DROP_IN="/etc/systemd/system/ollama.service.d/override.conf"
MODEL_TAG="llama3.2-vision:11b"

echo "=== Step 1: Stop ollama ==="
sudo systemctl stop ollama
echo "OK: stopped"

echo ""
echo "=== Step 2: Fix ownership and permissions ==="
sudo chown -R mhl:mhl "${OLLAMA_HOME}"
sudo chmod -R u+rwX "${OLLAMA_HOME}"
# Remove ACLs so User=ollama does not leave mixed permission masks
if command -v setfacl >/dev/null 2>&1; then
  sudo setfacl -R -b "${OLLAMA_HOME}" 2>/dev/null || true
fi
echo "OK: ${OLLAMA_HOME} owned by mhl:mhl"

echo ""
echo "=== Step 2b: Run service as mhl (required for mhl-owned models) ==="
sudo mkdir -p "$(dirname "${DROP_IN}")"
sudo tee "${DROP_IN}" >/dev/null <<EOF
[Service]
User=mhl
Group=mhl
Environment="HOME=/home/mhl"
Environment="OLLAMA_MODELS=${MODELS_DIR}"
EOF
echo "OK: systemd override → User=mhl, OLLAMA_MODELS=${MODELS_DIR}"

echo ""
echo "=== Step 3: Remove partial downloads ==="
PARTIAL_COUNT=$(find "${MODELS_DIR}/blobs" -name "*-partial-*" 2>/dev/null | wc -l)
if [[ "${PARTIAL_COUNT}" -gt 0 ]]; then
  find "${MODELS_DIR}/blobs" -name "*-partial-*" -delete
  echo "OK: removed ${PARTIAL_COUNT} partial blob(s)"
else
  echo "OK: no partial blobs found"
fi

echo ""
echo "=== Step 4: Start ollama ==="
sudo systemctl daemon-reload
sudo systemctl start ollama
echo "OK: started"

echo ""
echo "=== Step 5: Pull ${MODEL_TAG} ==="
sleep 3
ollama pull "${MODEL_TAG}"
echo "OK: pull finished"

echo ""
echo "=== Step 6: Test run ==="
ollama run "${MODEL_TAG}" "hi"
echo "OK: test run finished"

echo ""
echo "=== Step 7: Update SEA config ==="
mkdir -p "$(dirname "${SEA_CONFIG}")"
if [[ -f "${SEA_CONFIG}" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("${SEA_CONFIG}")
data = json.loads(p.read_text()) if p.exists() else {}
data["llmProvider"] = data.get("llmProvider") or "ollama"
data["ollamaModel"] = "${MODEL_TAG}"
p.write_text(json.dumps(data, indent=2) + "\n")
print("Updated", p)
PY
else
  cat > "${SEA_CONFIG}" <<EOF
{
  "llmProvider": "ollama",
  "ollamaModel": "${MODEL_TAG}"
}
EOF
fi
cat "${SEA_CONFIG}"
echo "OK: sea-config.json updated"

echo ""
echo "=== Done ==="
ollama list
