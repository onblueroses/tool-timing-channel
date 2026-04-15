#!/usr/bin/env bash
# Token-likelihood steganography experiments on Vast.ai GPU
# Expects VAST_API_KEY and VAST_INSTANCE_ID in environment
set -euo pipefail

echo "=== Setup ==="

# Install dependencies
pip install --quiet torch transformers accelerate hf_transfer
pip install --quiet numpy matplotlib

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Clone repo
git clone https://github.com/onblueroses/agent-stego.git /workspace/repo 2>/dev/null || true
cd /workspace/repo
pip install --quiet -e ".[token]"

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); assert torch.cuda.is_available()"

# Pre-download all three models
echo "=== Downloading models ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
for name in ['Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B']:
    print(f'Downloading {name}...')
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    print(f'  Done: {name}')
print('All models cached')
"

echo "=== Models ready, launching experiments ==="

# Write the experiment runner
cat > /workspace/run-experiments.sh << 'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
mkdir -p findings/token_likelihood

echo "$(date -Iseconds) Starting token-likelihood experiments"

# Phase 1: Single model test (3B, quick validation)
echo "=== Phase 1: Quick validation on 3B ==="
MODEL=Qwen/Qwen2.5-3B SECRET=HI TRIALS=3 MAX_TOKENS=50 \
  python3 experiments/token_likelihood.py 2>&1 | tee -a /workspace/run.log

# Phase 2: Full comparison across all models
echo "=== Phase 2: Multi-model comparison ==="
TRIALS=5 MAX_TOKENS=200 \
  python3 experiments/token_model_comparison.py 2>&1 | tee -a /workspace/run.log

# Phase 3: Detection experiment
echo "=== Phase 3: Detection baseline ==="
MODEL=Qwen/Qwen2.5-3B N=30 MAX_TOKENS=100 SECRET=HI \
  python3 experiments/token_detection.py 2>&1 | tee -a /workspace/run.log

# Also run detection on 7B
MODEL=Qwen/Qwen2.5-7B N=30 MAX_TOKENS=100 SECRET=HI \
  python3 experiments/token_detection.py 2>&1 | tee -a /workspace/run.log

echo "$(date -Iseconds) All experiments complete"
echo '{"status":"complete","timestamp":"'$(date -Iseconds)'"}' > /workspace/results/done.json
INNER

chmod +x /workspace/run-experiments.sh

# Write teardown wrapper
cat > /workspace/run-teardown.sh << EOF
#!/usr/bin/env bash
mkdir -p /workspace/results
bash /workspace/run-experiments.sh >> /workspace/run.log 2>&1

# Copy results to a known location for easy sync
cp -r /workspace/repo/findings/token_likelihood/* /workspace/results/ 2>/dev/null || true
cp /workspace/run.log /workspace/results/ 2>/dev/null || true

echo '{"status":"complete","timestamp":"'"\$(date -Iseconds)"'"}' > /workspace/results/done.json

# Self-destruct
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/?api_key=${VAST_API_KEY}" \
  || echo "WARNING: destroy failed" >> /workspace/run.log
EOF

chmod +x /workspace/run-teardown.sh
mkdir -p /workspace/results
nohup bash /workspace/run-teardown.sh </dev/null >> /workspace/run.log 2>&1 &
echo "PID: $! - SSH session done, experiments running"
echo "Monitor: vastai copy C.${VAST_INSTANCE_ID}:/workspace/run.log ./findings/vps/"
