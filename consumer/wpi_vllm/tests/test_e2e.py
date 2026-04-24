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

"""End-to-end test: Trainer → WPI Driver → vLLM weight update.

This test requires:
  - A running GKE cluster with WPI driver DaemonSet
  - vLLM server running with --weight-transfer-config '{"backend": "wpi"}'
  - Port-forwarding or direct network access to both

Usage:
    # From your local machine with kubectl configured:
    python test_e2e.py \
        --vllm-url http://localhost:8000 \
        --target-nodes 10.128.0.42 \
        --buffer-size-gb 1
"""

import argparse
import logging
import time

import requests
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_weight_update_roundtrip(vllm_url: str, target_nodes: list[str], buffer_size: int):
    """Full roundtrip test: init → pause → send weights → resume.

    Creates a small random model, sends its weights to vLLM via WPI,
    and verifies the HTTP responses at each stage.
    """
    from vllm.distributed.weight_transfer.wpi_engine import (
        WPITrainerSendWeightsArgs,
        WPIWeightTransferEngine,
    )

    # ── Step 1: Create a small test model ───────────────────────────────
    logger.info("Creating test model (2-layer MLP, ~4 MB)...")
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512, bias=True),
    ).cuda().to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.nbytes for p in model.parameters())
    logger.info("Model: %d params, %.2f MB", total_params, total_bytes / 1024**2)

    # ── Step 2: Initialize WPI on vLLM workers ──────────────────────────
    logger.info("Initializing WPI engine on vLLM workers...")
    resp = requests.post(
        f"{vllm_url}/init_weight_transfer_engine",
        json={
            "init_info": {
                "buffer_id": "e2e-test",
                "buffer_size_bytes": buffer_size,
                "socket_dir": "/run/wpi/sockets",
            }
        },
        timeout=120,
    )
    assert resp.status_code == 200, f"Init failed: {resp.text}"
    logger.info("✓ WPI engine initialized on vLLM")

    # ── Step 3: Initialize trainer-side context ─────────────────────────
    ctx = WPIWeightTransferEngine.trainer_init(
        dict(buffer_id="e2e-test", buffer_size_bytes=buffer_size),
        target_node_ids=target_nodes,
    )
    logger.info("✓ Trainer context initialized")

    # ── Step 4: Pause inference ─────────────────────────────────────────
    resp = requests.post(
        f"{vllm_url}/pause",
        params={"mode": "abort", "clear_cache": True},
        timeout=60,
    )
    assert resp.status_code == 200, f"Pause failed: {resp.text}"
    logger.info("✓ vLLM inference paused")

    # Verify paused
    resp = requests.get(f"{vllm_url}/is_paused", timeout=10)
    assert resp.json()["is_paused"], "Expected paused=True"
    logger.info("✓ Confirmed paused")

    # ── Step 5: Send weights ────────────────────────────────────────────
    logger.info("Sending weights via WPI...")
    send_start = time.time()

    param_iter = (
        (name, param)
        for name, param in model.named_parameters()
    )
    send_args = WPITrainerSendWeightsArgs(
        mode="http",
        url=vllm_url,
        trainer_ctx=ctx,
    )
    WPIWeightTransferEngine.trainer_send_weights(param_iter, send_args)

    send_elapsed = time.time() - send_start
    bandwidth = (total_bytes / 1024**2) / send_elapsed if send_elapsed > 0 else 0
    logger.info(
        "✓ Weights sent: %.2f MB in %.3fs (%.1f MB/s)",
        total_bytes / 1024**2,
        send_elapsed,
        bandwidth,
    )

    # ── Step 6: Resume inference ────────────────────────────────────────
    resp = requests.post(f"{vllm_url}/resume", timeout=60)
    assert resp.status_code == 200, f"Resume failed: {resp.text}"
    logger.info("✓ vLLM inference resumed")

    # ── Step 7: Verify vLLM is serving ──────────────────────────────────
    resp = requests.get(f"{vllm_url}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    logger.info("✓ vLLM is healthy after weight update")

    logger.info("═══ E2E TEST PASSED ═══")


def main():
    parser = argparse.ArgumentParser(description="WPI + vLLM E2E test")
    parser.add_argument("--vllm-url", required=True, help="vLLM server base URL")
    parser.add_argument(
        "--target-nodes",
        required=True,
        help="Comma-separated list of vLLM node IPs",
    )
    parser.add_argument(
        "--buffer-size-gb",
        type=float,
        default=1.0,
        help="WPI buffer size in GiB",
    )
    args = parser.parse_args()

    target_nodes = [n.strip() for n in args.target_nodes.split(",")]
    buffer_size = int(args.buffer_size_gb * 1024**3)

    test_weight_update_roundtrip(args.vllm_url, target_nodes, buffer_size)


if __name__ == "__main__":
    main()
