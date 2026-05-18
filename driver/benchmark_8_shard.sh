#!/bin/bash
set -e

echo "Applying sharded test CRDs for 8 shards..."
kubectl apply -f ../crds/weightbuffer.yaml
kubectl apply -f ../crds/weightclaim.yaml
kubectl apply -f ../crds/test_wb_sharded.yaml

# Generate 16 WeightClaims: 8 source-side + 8 target-side
# Each shard (0-7) needs to be staged on BOTH nodes
for i in {0..7}; do
cat << WC_EOF | kubectl apply -f -
apiVersion: wpi.io/v1alpha1
kind: WeightClaim
metadata:
  name: wc-test-sharded-src-${i}
  namespace: wpi-system
spec:
  weightBufferName: wb-test-sharded
  shardIndex: ${i}
WC_EOF
cat << WC_EOF | kubectl apply -f -
apiVersion: wpi.io/v1alpha1
kind: WeightClaim
metadata:
  name: wc-test-sharded-tgt-${i}
  namespace: wpi-system
spec:
  weightBufferName: wb-test-sharded
  shardIndex: ${i}
WC_EOF
done

echo "Cleaning up any existing benchmark pods..."
kubectl delete pod -l benchmark=8shard -n wpi-system --ignore-not-found || true

kubectl apply -f ../crds/dra_resources.yaml

# Wait for driver pods to be Ready
echo "Waiting for WPI driver pods to be Ready before port-forwarding..."
kubectl wait --for=condition=Ready pod -l app=wpi-driver -n wpi-system --timeout=120s

# Run the 8-shard Python benchmark wrapper
python3 test_scatter_8_shards.py
