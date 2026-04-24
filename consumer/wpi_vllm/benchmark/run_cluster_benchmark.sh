#!/bin/bash
# Copyright 2026 Google LLC
#
# Automates running WPI + vLLM benchmark inside the cluster.
#
# This script:
#   1. Creates a ConfigMap with the benchmark code.
#   2. Discovers the vLLM pod's host IP.
#   3. Modifies trainer.yaml with the host IP and applies it.
#
# Usage:
#   ./run_cluster_benchmark.sh [model_name]
#
# Example:
#   ./run_cluster_benchmark.sh Qwen/Qwen2-7B

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen2-7B}"
NAMESPACE="wpi-system"

echo "══════════════════════════════════════════════════════════════"
echo "WPI + vLLM Cluster Benchmark Runner"
echo "══════════════════════════════════════════════════════════════"
echo "Model: ${MODEL_NAME}"
echo "Namespace: ${NAMESPACE}"
echo ""

echo "1. Creating ConfigMap for trainer code..."
kubectl create configmap wpi-vllm-trainer-code \
    --from-file=bench_weight_update.py \
    -n ${NAMESPACE} \
    --dry-run=client -o yaml | kubectl apply -f -

echo "2. Discovering vLLM pod..."
VLLM_POD=$(kubectl get pod -l app=vllm-wpi -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
echo "Found vLLM pod: ${VLLM_POD}"

echo "3. Discovering target node IP..."
TARGET_NODE_IP=$(kubectl get pod ${VLLM_POD} -n ${NAMESPACE} -o jsonpath='{.status.hostIP}')
echo "Target node IP: ${TARGET_NODE_IP}"

echo "4. Generating modified trainer manifest..."
# We use a temp file to avoid modifying the source trainer.yaml file
# We replace the empty TARGET_NODE_IPS value with the actual IP
sed "s/value: \"\"/value: \"${TARGET_NODE_IP}\"/" ../deploy/trainer.yaml > /tmp/trainer_modified.yaml

# Also update the model name if it was specified
if [ "$MODEL_NAME" != "meta-llama/Llama-3-8B" ]; then
    sed -i "s#value: \"Qwen/Qwen2-7B\"#value: \"${MODEL_NAME}\"#" /tmp/trainer_modified.yaml
fi

echo "5. Deploying trainer pod..."
kubectl apply -f /tmp/trainer_modified.yaml -n ${NAMESPACE}

echo ""
echo "✅ Benchmark triggered successfully!"
echo "You can monitor the progress and see results with:"
echo "  kubectl logs wpi-vllm-trainer -n ${NAMESPACE}"
echo ""
echo "Note: To clean up after the benchmark is done, delete the pod:"
echo "  kubectl delete pod wpi-vllm-trainer -n ${NAMESPACE}"
