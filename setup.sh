#!/bin/bash

# WPI (Weight Pooling Interface) Installation & Startup Script
# This script deploys all WPI components from scratch to a Kubernetes cluster.

set -e

echo "Starting WPI Installation..."
echo ""

echo "0. Configuring IAM permissions for Artifact Registry access..."
CONTEXT=$(kubectl config current-context)
if [[ "$CONTEXT" == gke_* ]]; then
    PROJECT_ID=$(echo "$CONTEXT" | awk -F_ '{print $2}')
    LOCATION=$(echo "$CONTEXT" | awk -F_ '{print $3}')
    CLUSTER_NAME=$(echo "$CONTEXT" | awk -F_ '{print $4}')
    
    echo "Detected GKE cluster: $CLUSTER_NAME in $LOCATION (Project: $PROJECT_ID)"
    SAs=$(gcloud container node-pools list --cluster="$CLUSTER_NAME" --location="$LOCATION" --format="value(config.serviceAccount)" || true)
    
    for SA in $SAs; do
        if [ "$SA" != "default" ] && [ -n "$SA" ]; then
            echo "Granting roles/artifactregistry.reader to: $SA"
            gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$SA" \
                --role="roles/artifactregistry.reader" \
                --condition=None >/dev/null 2>&1 || true
        fi
    done
else
    echo "Not a GKE context ($CONTEXT), skipping IAM permission setup."
fi
echo ""
echo "1. Creating WPI namespace..."
kubectl create namespace wpi-system --dry-run=client -o yaml | kubectl apply -f -

echo "2. Installing Custom Resource Definitions (CRDs)..."
# Apply platform CRDs first (excluding test/mock CRDs like test_wb.yaml)
kubectl apply -f crds/weightbuffer.yaml
kubectl apply -f crds/weightclaim.yaml

echo "3. Creating WPI Runtime ConfigMap..."
# Fix permissions on generated protobuf files which may be owned by root
if [ -f "driver/wpi_pb2.py" ]; then
    sudo chown $USER:$(id -g) driver/wpi_pb2*.py || true
fi

# The wpi-driver DaemonSet mounts this script dynamically over the container image
kubectl create configmap wpi-code \
    --from-file=driver/main.py \
    --from-file=driver/wpi_pb2.py \
    --from-file=driver/wpi_pb2_grpc.py \
    -n wpi-system \
    --dry-run=client -o yaml | kubectl apply -f -

echo "4. Deploying the WPI Driver DaemonSet..."
kubectl apply -f driver/daemonset.yaml

echo "5. Deploying the WPI Operator..."
if [ -d "operator/config/default" ]; then
    echo "Applying Operator via Kustomize..."
    kubectl apply -k operator/config/default || echo "Warning: Kustomize failed. Ensure you have the operator manifests correctly set up."
else
    echo "Warning: Operator config directory not found. Skipping Operator deployment."
fi

echo "6. Waiting for the WPI Driver hardware rollout..."
kubectl rollout status daemonset wpi-driver -n wpi-system --timeout=120s

echo ""
echo "✅ WPI Installation Complete!"
echo "WPI is now running on your cluster."
echo ""
echo "Next steps for the Demo:"
echo "  1. Apply your mock WeightBuffers: kubectl apply -f crds/test_wb.yaml && kubectl apply -f crds/test_wc.yaml"
echo "  2. Run the Benchmark script:      cd driver && ./benchmark.sh"
echo "  3. Deploy your Inference Pod:     kubectl apply -f driver/demo_inference_pod.yaml"
