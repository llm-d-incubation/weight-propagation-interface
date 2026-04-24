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

"""Example: Trainer sending weights to vLLM via HTTP + WPI.

This script demonstrates the full RLHF weight sync loop:
  1. Train a model (simulated with random weight perturbation)
  2. Pause vLLM inference
  3. Pack weights into the WPI VRAM buffer
  4. Trigger NodePropagate (NCCL broadcast via WPI driver)
  5. Send offset metadata to vLLM via HTTP /update_weights
  6. Resume vLLM inference

Usage:
    python trainer_http.py \
        --vllm-url http://vllm-server:8000 \
        --target-nodes 10.0.0.2,10.0.0.3 \
        --model meta-llama/Llama-3-8B \
        --buffer-size-gb 20 \
        --num-steps 10
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


def compute_buffer_size(model: torch.nn.Module, headroom: float = 1.05) -> int:
    """Compute required WPI buffer size from model parameters."""
    total = sum(p.nbytes for p in model.parameters())
    return int(total * headroom)


def main():
    parser = argparse.ArgumentParser(description="WPI + vLLM trainer (HTTP mode)")
    parser.add_argument("--vllm-url", required=True, help="vLLM server base URL")
    parser.add_argument(
        "--target-nodes",
        required=True,
        help="Comma-separated list of vLLM node IPs for NodePropagate",
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3-8B", help="Model name/path"
    )
    parser.add_argument(
        "--buffer-size-gb",
        type=float,
        default=0,
        help="WPI buffer size in GiB (0 = auto-detect from model)",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="Number of training steps")
    parser.add_argument(
        "--socket-dir",
        default="/run/wpi/sockets",
        help="WPI UNIX socket directory",
    )
    parser.add_argument("--buffer-id", default="vllm-weights", help="WPI buffer ID")
    args = parser.parse_args()

    target_nodes = [n.strip() for n in args.target_nodes.split(",")]

    # ── Load model ──────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    buffer_size = (
        int(args.buffer_size_gb * 1024**3)
        if args.buffer_size_gb > 0
        else compute_buffer_size(model)
    )
    logger.info("Buffer size: %.2f GiB", buffer_size / 1024**3)

    # ── Initialize WPI engine on vLLM workers ───────────────────────────
    logger.info("Initializing WPI engine on vLLM workers...")
    resp = requests.post(
        f"{args.vllm_url}/init_weight_transfer_engine",
        json={
            "init_info": {
                "buffer_id": args.buffer_id,
                "buffer_size_bytes": buffer_size,
                "socket_dir": args.socket_dir,
            }
        },
        timeout=120,
    )
    resp.raise_for_status()
    logger.info("vLLM WPI engine initialized: %s", resp.json())

    # ── Initialize trainer-side WPI context ─────────────────────────────
    from vllm.distributed.weight_transfer.wpi_engine import (
        WPITrainerSendWeightsArgs,
        WPIWeightTransferEngine,
    )

    ctx = WPIWeightTransferEngine.trainer_init(
        dict(
            buffer_id=args.buffer_id,
            buffer_size_bytes=buffer_size,
            socket_dir=args.socket_dir,
        ),
        target_node_ids=target_nodes,
    )
    logger.info("Trainer WPI context initialized.")

    # ── Training loop ───────────────────────────────────────────────────
    for step in range(args.num_steps):
        step_start = time.time()
        logger.info("═══ Step %d/%d ═══", step + 1, args.num_steps)

        # Simulate training: perturb weights slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.001)

        # ── Pause vLLM ──────────────────────────────────────────────────
        logger.info("Pausing vLLM inference...")
        resp = requests.post(
            f"{args.vllm_url}/pause",
            params={"mode": "abort", "clear_cache": True},
            timeout=60,
        )
        resp.raise_for_status()

        # ── Send weights ────────────────────────────────────────────────
        logger.info("Sending weights via WPI...")
        send_start = time.time()

        param_iter = ((n, p) for n, p in model.named_parameters())
        send_args = WPITrainerSendWeightsArgs(
            mode="http",
            url=args.vllm_url,
            trainer_ctx=ctx,
        )
        WPIWeightTransferEngine.trainer_send_weights(param_iter, send_args)

        send_elapsed = time.time() - send_start
        total_bytes = sum(p.nbytes for p in model.parameters())
        bandwidth = (total_bytes / 1024**3) / send_elapsed if send_elapsed > 0 else 0
        logger.info(
            "Weight sync: %.1f MB in %.2fs (%.2f GB/s)",
            total_bytes / 1024**2,
            send_elapsed,
            bandwidth,
        )

        # ── Resume vLLM ─────────────────────────────────────────────────
        logger.info("Resuming vLLM inference...")
        resp = requests.post(f"{args.vllm_url}/resume", timeout=60)
        resp.raise_for_status()

        step_elapsed = time.time() - step_start
        logger.info("Step %d complete in %.2fs", step + 1, step_elapsed)

    logger.info("Training complete. %d steps executed.", args.num_steps)


if __name__ == "__main__":
    main()
