#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Orchestrates WPI + vLLM benchmark on a GKE cluster.
#
# Prerequisites:
#   - kubectl configured for your GKE cluster
#   - WPI driver DaemonSet running in wpi-system namespace
#   - vLLM pod deployed with --weight-transfer-config '{"backend": "wpi"}'
#
# Usage:
#   bash run_benchmark.sh [model_name] [num_iters]
#
# Example:
#   bash run_benchmark.sh meta-llama/Llama-3-8B 20

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3-8B}"
NUM_ITERS="${2:-20}"
WARMUP="${3:-3}"
NAMESPACE="wpi-system"
VLLM_PORT=8000
OUTPUT_FILE="bench_results_$(date +%Y%m%d_%H%M%S).json"

echo "══════════════════════════════════════════════════════════════"
echo "WPI + vLLM Weight Update Benchmark"
echo "══════════════════════════════════════════════════════════════"
echo "Model:     ${MODEL}"
echo "Iters:     ${NUM_ITERS} (+ ${WARMUP} warmup)"
echo "Namespace: ${NAMESPACE}"
echo "Output:    ${OUTPUT_FILE}"
echo ""

# ── Step 1: Apply WPI resources ─────────────────────────────────────────
echo "Step 1: Applying WPI resources..."
kubectl apply -f ../deploy/wpi_resources.yaml -n ${NAMESPACE}
sleep 2

# ── Step 2: Wait for vLLM pod ──────────────────────────────────────────
echo "Step 2: Waiting for vLLM server pod..."
kubectl wait --for=condition=Ready pod -l app=vllm-wpi -n ${NAMESPACE} --timeout=300s

VLLM_POD=$(kubectl get pod -l app=vllm-wpi -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
echo "vLLM pod: ${VLLM_POD}"

# ── Step 3: Get target node IPs ─────────────────────────────────────────
echo "Step 3: Discovering target node IPs..."
TARGET_NODES=$(kubectl get pod ${VLLM_POD} -n ${NAMESPACE} -o jsonpath='{.status.hostIP}')
echo "Target nodes: ${TARGET_NODES}"

# ── Step 4: Port-forward to vLLM ────────────────────────────────────────
echo "Step 4: Port-forwarding to vLLM server..."
kubectl port-forward pod/${VLLM_POD} ${VLLM_PORT}:${VLLM_PORT} -n ${NAMESPACE} > /dev/null 2>&1 &
PF_PID=$!
trap "kill ${PF_PID} 2>/dev/null" EXIT INT TERM

# Wait for port-forward
for i in $(seq 1 15); do
    if nc -z localhost ${VLLM_PORT} 2>/dev/null; then
        echo "Port-forward ready!"
        break
    fi
    sleep 1
done

# ── Step 5: Verify vLLM health ──────────────────────────────────────────
echo "Step 5: Checking vLLM health..."
HEALTHY=false
for i in $(seq 1 60); do
    if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null; then
        echo "vLLM is healthy."
        HEALTHY=true
        break
    fi
    echo "Waiting for vLLM to become healthy (attempt $i)..."
    sleep 5
done

if [ "$HEALTHY" = false ]; then
    echo "Error: vLLM did not become healthy in time."
    exit 1
fi

# ── Step 6: Run benchmark ──────────────────────────────────────────────
echo ""
echo "Step 6: Running benchmark..."
echo ""

python bench_weight_update.py \
    --vllm-url http://localhost:${VLLM_PORT} \
    --target-nodes "${TARGET_NODES}" \
    --model "${MODEL}" \
    --num-iters "${NUM_ITERS}" \
    --warmup-iters "${WARMUP}" \
    --output "${OUTPUT_FILE}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Benchmark complete. Results: ${OUTPUT_FILE}"
echo "══════════════════════════════════════════════════════════════"
