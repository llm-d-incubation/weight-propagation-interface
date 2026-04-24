# WPI ↔ vLLM Integration

This directory contains everything needed to run vLLM with the WPI weight transfer backend for RLHF and online training workflows.

## Directory Structure

```
wpi_vllm/
├── README.md               # This file
├── deploy/
│   ├── vllm_server.yaml    # K8s pod spec for vLLM serving with WPI
│   ├── trainer.yaml         # K8s pod spec for the trainer
│   └── wpi_resources.yaml   # WeightBuffer + WeightClaim CRDs
├── examples/
│   ├── trainer_http.py      # Trainer → vLLM via HTTP /update_weights
│   └── trainer_ray.py       # Trainer → vLLM via Ray handle
├── tests/
│   ├── test_wpi_engine.py   # Unit tests for the vLLM WPI engine
│   └── test_e2e.py          # E2E test: trainer → WPI driver → vLLM
└── benchmark/
    ├── bench_weight_update.py   # Benchmark: weight update latency & throughput
    └── run_benchmark.sh         # Orchestrates deploy + bench on GKE
```

## Prerequisites

1. **WPI driver** DaemonSet running on GPU nodes (`wpi-system` namespace)
2. **vLLM** built from the WPI-enabled fork (with `wpi_engine.py`)
3. **`wpi_client`** installed in the vLLM container:
   ```bash
   pip install wpi_client   # core client only (no verl/ray dependency)
   ```

## Quick Start

### 1. Deploy WPI resources

```bash
kubectl apply -f deploy/wpi_resources.yaml
```

### 2. Start vLLM server with WPI backend

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3-8B \
  --weight-transfer-config '{"backend": "wpi"}' \
  --gpu-memory-utilization 0.8
```

Or deploy on K8s:
```bash
kubectl apply -f deploy/vllm_server.yaml
```

### 3. Run the trainer

```bash
# HTTP mode (standalone)
python examples/trainer_http.py \
  --vllm-url http://vllm-server:8000 \
  --target-nodes 10.0.0.2

# Ray mode (co-located with Ray cluster)
python examples/trainer_ray.py
```

### 4. Benchmark

```bash
bash benchmark/run_benchmark.sh
```

## Architecture

```
Trainer Node                         vLLM Inference Node
┌────────────────────┐               ┌────────────────────────┐
│  trainer_http.py   │               │  vLLM Server           │
│  ┌──────────────┐  │               │  (--backend wpi)       │
│  │ Pack weights  │  │               │                        │
│  │ into VRAM buf │  │  NodePropagate│  ┌──────────────────┐  │
│  │      ↓        │  │  (NCCL bcast) │  │ WPI VRAM Buffer  │  │
│  │ WPI propagate │──┼──────────────→│  │ (persistent)     │  │
│  │      ↓        │  │               │  └────────┬─────────┘  │
│  │ HTTP metadata │──┼──/update_wts─→│           ↓            │
│  └──────────────┘  │               │  receive_weights()      │
│                    │               │   → wait_for_ready()    │
│  WPI Driver        │               │   → unpack from buffer  │
│  (DaemonSet)       │               │   → model.load_weights()│
└────────────────────┘               │                        │
                                     │  WPI Driver             │
                                     │  (DaemonSet)            │
                                     └────────────────────────┘
```
