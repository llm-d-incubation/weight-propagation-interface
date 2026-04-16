#!/bin/bash
set -e

echo "Applying sharded test CRDs..."
kubectl apply -f ../crds/weightbuffer.yaml
kubectl apply -f ../crds/weightclaim.yaml
kubectl apply -f ../crds/test_wb_sharded.yaml -n wpi-system
kubectl apply -f ../crds/test_wc_sharded_0.yaml -n wpi-system
kubectl apply -f ../crds/test_wc_sharded_1.yaml -n wpi-system
kubectl apply -f ../crds/dra_resources.yaml

# Wait for driver to reconcile
sleep 2

echo "Waiting for WPI driver pods to be Ready before port-forwarding..."
kubectl wait --for=condition=Ready pod -l app=wpi-driver -n wpi-system --timeout=120s

PODS=($(kubectl get pods -n wpi-system -l app=wpi-driver -o jsonpath='{.items[*].metadata.name}'))
echo "Source Pod: ${PODS[0]} Target Pod: ${PODS[1]}"

kubectl port-forward pods/${PODS[0]} 50051:50051 -n wpi-system > /dev/null 2>&1 &
PF1_PID=$!

kubectl port-forward pods/${PODS[1]} 50061:50051 -n wpi-system > /dev/null 2>&1 &
PF2_PID=$!

trap "kill $PF1_PID $PF2_PID 2>/dev/null" EXIT INT TERM

echo "Waiting for port-forwards to establish..."
for i in {1..20}; do
  if nc -z localhost 50051 && nc -z localhost 50061; then
    echo "Ports are up! Executing scatter benchmark..."
    break
  fi
  sleep 1
done

python3 test_scatter_benchmark.py
