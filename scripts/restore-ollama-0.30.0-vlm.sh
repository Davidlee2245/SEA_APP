#!/usr/bin/env bash
# Restore Ollama v0.30.0 and pull RTX 5090–compatible VLMs (llava, gemma4, qwen3-vl:32b).
# Usage: bash scripts/restore-ollama-0.30.0-vlm.sh

set -euo pipefail

VERSION="0.30.0"
DROP_IN="/etc/systemd/system/ollama.service.d/override.conf"
SEA_CONFIG="/home/mhl/.config/sea-exosome-analysis/sea-config.json"
MODELS_ROOT="/home/mhl/.ollama/models"
VISION_MANIFEST="${MODELS_ROOT}/manifests/registry.ollama.ai/library/llama3.2-vision"
MODELS=(llava gemma4 "qwen3-vl:32b")

echo "=== Step 1: Stop ollama ==="
sudo systemctl stop ollama
echo "OK"

echo ""
echo "=== Step 2: Install Ollama ${VERSION} (official install.sh) ==="
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="${VERSION}" sh
# Remove pinned downgrades from earlier troubleshooting (v0.5.7 / v0.22.1, etc.)
sudo rm -rf /usr/share/ollama-v* 2>/dev/null || true
echo "OK"

echo ""
echo "=== Step 3: systemd override (no LD_LIBRARY_PATH) ==="
sudo mkdir -p "$(dirname "${DROP_IN}")"
sudo tee "${DROP_IN}" >/dev/null <<'EOF'
[Service]
User=mhl
Group=mhl
Environment="HOME=/home/mhl"
Environment="OLLAMA_MODELS=/home/mhl/.ollama/models"
EOF
cat "${DROP_IN}"
echo "OK"

echo ""
echo "=== Step 4: Start ollama ==="
sudo systemctl daemon-reload
sudo systemctl start ollama
sleep 3
echo "OK"

echo ""
echo "=== Step 5: Verify version ==="
ollama --version
curl -s http://127.0.0.1:11434/api/version
echo ""

echo ""
echo "=== Step 6: Clean broken llama3.2-vision / partial blobs ==="
rm -rf "${VISION_MANIFEST}" 2>/dev/null || sudo rm -rf "${VISION_MANIFEST}"
find "${MODELS_ROOT}/blobs" -name "*-partial-*" -delete 2>/dev/null || true
echo "OK"

echo ""
echo "=== Step 7: Pull and test VLMs ==="
for model in "${MODELS[@]}"; do
  echo "--- pull ${model} ---"
  ollama pull "${model}"
  echo "--- run ${model} ---"
  ollama run "${model}" "hi" || echo "ERROR: run ${model} failed"
  echo ""
done

echo ""
echo "=== Step 8: ollama list ==="
ollama list

echo ""
echo "=== Step 9: Update sea-config.json ==="
mkdir -p "$(dirname "${SEA_CONFIG}")"
python3 - <<PY
import json
from pathlib import Path
p = Path("${SEA_CONFIG}")
data = json.loads(p.read_text()) if p.exists() else {}
data["llmProvider"] = "ollama"
data["ollamaModel"] = "gemma4"
p.write_text(json.dumps(data, indent=2) + "\n")
print(p.read_text())
PY

echo ""
echo "=== Done ==="
