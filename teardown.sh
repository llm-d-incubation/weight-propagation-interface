#!/bin/bash

# WPI (Weight Pooling Interface) Cleanup & Teardown Script
# This script removes all WPI components, CRDs, daemonsets, and test resources from the Kubernetes cluster.

echo "Starting WPI Teardown..."

echo "1. Deleting Active Demo/Test Resources..."
kubectl delete -f driver/demo_inference_pod.yaml --ignore-not-found
kubectl delete -f crds/test_wc.yaml --ignore-not-found
kubectl delete -f crds/test_wb.yaml --ignore-not-found

# Also try to clean up any dynamically created WeightClaims/WeightBuffers/ResourceClaims
echo "Cleaning up lingering resources across all namespaces..."
kubectl delete weightclaims --all --all-namespaces --ignore-not-found >/dev/null 2>&1 || true
kubectl delete weightbuffers --all --all-namespaces --ignore-not-found >/dev/null 2>&1 || true
kubectl delete resourceclaims --all --all-namespaces --ignore-not-found >/dev/null 2>&1 || true

echo "2. Deleting WPI Driver & Runtime Configs..."
kubectl delete -f driver/daemonset.yaml --ignore-not-found
kubectl delete configmap wpi-code -n wpi-system --ignore-not-found

echo "3. Deleting WPI Operator..."
if [ -d "operator" ]; then
    kubectl delete -f operator/ --ignore-not-found
fi

echo "4. Deleting Custom Resource Definitions (CRDs)..."
kubectl delete -f crds/dra_resources.yaml --ignore-not-found >/dev/null 2>&1 || true
kubectl delete -f crds/weightbuffer.yaml --ignore-not-found >/dev/null 2>&1 || true
kubectl delete -f crds/weightclaim.yaml --ignore-not-found >/dev/null 2>&1 || true

echo "5. Deleting the WPI Namespace..."
kubectl delete namespace wpi-system --ignore-not-found

echo "✅ WPI Teardown Complete."
