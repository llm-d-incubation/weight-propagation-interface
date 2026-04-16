#!/bin/bash
# deploy_sharding_update.sh
# Deploys the WPI sharding update to the GKE cluster.
# 
# Prerequisites:
#   - kubectl configured with the correct GKE cluster context
#   - gcloud authenticated with artifact registry access
#   - docker installed (only needed for image rebuild)
#
# Usage:
#   ./deploy_sharding_update.sh              # ConfigMap-only update (fast, no rebuild)
#   ./deploy_sharding_update.sh --rebuild    # Full Docker image rebuild + push + deploy

set -e

WPI_DIR="$(cd "$(dirname "$0")" && pwd)"
NAMESPACE="wpi-system"

echo "=== WPI Sharding Update Deploy ==="
echo ""

# Step 1: Update CRDs with new sharding fields
echo "1. Updating CRDs..."
kubectl apply -f "${WPI_DIR}/crds/weightbuffer.yaml"
kubectl apply -f "${WPI_DIR}/crds/weightclaim.yaml"
echo "   CRDs updated."

# Step 2: Update ConfigMap with new driver code + proto stubs
echo "2. Updating wpi-code ConfigMap..."
kubectl create configmap wpi-code \
    --from-file="${WPI_DIR}/driver/main.py" \
    --from-file="${WPI_DIR}/driver/wpi_pb2.py" \
    --from-file="${WPI_DIR}/driver/wpi_pb2_grpc.py" \
    -n "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
echo "   ConfigMap updated."

# Step 3: Optionally rebuild and push Docker image
if [ "$1" = "--rebuild" ]; then
    echo "3. Rebuilding Docker image..."
    
    # Try the build_plugin.sh registry first, fall back to daemonset registry
    # REGISTRY="us-west1-docker.pkg.dev/gke-shared-ai-dev/rl-weight-transfer/wpi-driver:latest"
    REGISTRY="us-central1-docker.pkg.dev/yangspirit-gke-dev/rl-weight-transfer/wpi-driver:latest"
    
    docker build --platform linux/amd64 -f "${WPI_DIR}/driver/Dockerfile" -t "${REGISTRY}" "${WPI_DIR}"
    docker push "${REGISTRY}"
    echo "   Image pushed to ${REGISTRY}"
else
    echo "3. Skipping Docker rebuild (use --rebuild to include)."
fi

# Step 4: Rolling restart to pick up ConfigMap changes
echo "4. Restarting wpi-driver DaemonSet..."
kubectl rollout restart daemonset wpi-driver -n "${NAMESPACE}"
kubectl rollout status daemonset wpi-driver -n "${NAMESPACE}" --timeout=120s
echo "   DaemonSet restarted."

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Verify with:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo "  kubectl logs -n ${NAMESPACE} -l app=wpi-driver --tail=20"
echo ""
echo "New capabilities deployed:"
echo "  - WeightBuffer sharding (TensorParallel, ExpertParallel, PipelineParallel, Custom)"
echo "  - WeightClaim shard auto-assignment (from pod annotations)"
echo "  - NodeStageWeight shard-scoped buffers (buffer_id__shard_N)"
echo "  - NodePropagate SCATTER mode (ncclSend/ncclRecv per-target)"
echo "  - PRE_UPDATE consumer notifications"
