#!/bin/bash
set -e
docker build --platform linux/amd64 -t us-west1-docker.pkg.dev/gke-shared-ai-dev/rl-weight-transfer/wpi-driver:latest -f Dockerfile ..
docker push us-west1-docker.pkg.dev/gke-shared-ai-dev/rl-weight-transfer/wpi-driver:latest
kubectl rollout restart daemonset wpi-driver -n wpi-system
