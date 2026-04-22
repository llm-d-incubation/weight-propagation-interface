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

"""Example: Trainer sending weights to vLLM via Ray + WPI.

Same as trainer_http.py but uses Ray RPC to call vLLM's update_weights()
instead of HTTP. This is the preferred mode when trainer and vLLM are
co-located in the same Ray cluster (e.g., verl-style RLHF).

Usage:
    python trainer_ray.py \
        --target-nodes 10.0.0.2,10.0.0.3 \
        --model meta-llama/Llama-3-8B \
        --num-steps 5
"""

import argparse
import logging
import time

import ray
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="WPI + vLLM trainer (Ray mode)")
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
    parser.add_argument("--buffer-id", default="vllm-weights", help="WPI buffer ID")
    parser.add_argument(
        "--socket-dir",
        default="/run/wpi/sockets",
        help="WPI UNIX socket directory",
    )
    args = parser.parse_args()

    target_nodes = [n.strip() for n in args.target_nodes.split(",")]

    # ── Initialize Ray ──────────────────────────────────────────────────
    ray.init()

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
        else int(sum(p.nbytes for p in model.parameters()) * 1.05)
    )
    logger.info("Buffer size: %.2f GiB", buffer_size / 1024**3)

    # ── Get vLLM Ray handle ─────────────────────────────────────────────
    # In a real verl setup, you'd get this from the rollout worker group.
    # Here we assume a Ray-based vLLM actor is already running.
    from vllm.entrypoints.llm import LLM

    llm_handle = ray.get_actor("vllm_server")  # Named Ray actor

    # ── Initialize WPI engine on vLLM workers ───────────────────────────
    from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

    ray.get(
        llm_handle.init_weight_transfer_engine.remote(
            WeightTransferInitRequest(
                init_info={
                    "buffer_id": args.buffer_id,
                    "buffer_size_bytes": buffer_size,
                    "socket_dir": args.socket_dir,
                }
            )
        )
    )
    logger.info("vLLM WPI engine initialized via Ray.")

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

    # ── Training loop ───────────────────────────────────────────────────
    for step in range(args.num_steps):
        step_start = time.time()
        logger.info("═══ Step %d/%d ═══", step + 1, args.num_steps)

        # Simulate training
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.001)

        # Pause, send, resume via Ray
        ray.get(llm_handle.pause_generation.remote(mode="abort", clear_cache=True))

        param_iter = ((n, p) for n, p in model.named_parameters())
        send_args = WPITrainerSendWeightsArgs(
            mode="ray",
            llm_handle=llm_handle,
            trainer_ctx=ctx,
        )
        WPIWeightTransferEngine.trainer_send_weights(param_iter, send_args)

        ray.get(llm_handle.resume_generation.remote())

        step_elapsed = time.time() - step_start
        logger.info("Step %d complete in %.2fs", step + 1, step_elapsed)

    logger.info("Training complete. %d steps executed.", args.num_steps)
    ray.shutdown()


if __name__ == "__main__":
    main()
