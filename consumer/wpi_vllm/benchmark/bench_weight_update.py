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

"""Benchmark: WPI weight update latency and throughput.

Measures end-to-end weight update performance across multiple iterations:
  - Pack time (memcpy into WPI buffer)
  - Propagate time (NCCL broadcast via WPI driver)
  - HTTP metadata time
  - Total wall time
  - Effective bandwidth (GB/s)

Usage:
    python bench_weight_update.py \
        --vllm-url http://vllm-server:8000 \
        --target-nodes 10.0.0.2 \
        --model meta-llama/Llama-3-8B \
        --num-iters 20 \
        --warmup-iters 3
"""

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass

import requests
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    num_params: int
    total_bytes: int
    num_iters: int

    # Per-iteration timings (seconds)
    pack_times: list[float]
    propagate_times: list[float]
    metadata_times: list[float]
    total_times: list[float]

    def summary(self) -> dict:
        """Return summary statistics."""
        def stats(values):
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": statistics.median(values),
                "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
                "min": min(values),
                "max": max(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            }

        size_gb = self.total_bytes / 1024**3
        bw_values = [size_gb / t for t in self.total_times if t > 0]

        return {
            "model": self.model_name,
            "params": f"{self.num_params / 1e6:.1f}M",
            "weight_size_gb": f"{size_gb:.3f}",
            "iters": self.num_iters,
            "pack_ms": {k: v * 1000 for k, v in stats(self.pack_times).items()},
            "propagate_ms": {k: v * 1000 for k, v in stats(self.propagate_times).items()},
            "metadata_ms": {k: v * 1000 for k, v in stats(self.metadata_times).items()},
            "total_ms": {k: v * 1000 for k, v in stats(self.total_times).items()},
            "bandwidth_gbs": stats(bw_values),
        }


def run_benchmark(
    vllm_url: str,
    target_nodes: list[str],
    model_name: str,
    buffer_size_gb: float,
    num_iters: int,
    warmup_iters: int,
) -> BenchmarkResult:
    """Run the weight update benchmark."""
    from vllm.distributed.weight_transfer.wpi_engine import (
        WPITrainerSendWeightsArgs,
        WPITrainerContext,
        WPIWeightTransferEngine,
        WPIWeightTransferUpdateInfo,
    )
    import math
    from dataclasses import asdict as _asdict

    # ── Load model ──────────────────────────────────────────────────────
    logger.info("Loading model: %s", model_name)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.nbytes for p in model.parameters())
    buffer_size = (
        int(buffer_size_gb * 1024**3)
        if buffer_size_gb > 0
        else int(total_bytes * 1.05)
    )
    logger.info(
        "Model: %d params, %.2f GB weights, %.2f GB buffer",
        num_params,
        total_bytes / 1024**3,
        buffer_size / 1024**3,
    )

    # ── Init engines ────────────────────────────────────────────────────
    resp = requests.post(
        f"{vllm_url}/init_weight_transfer_engine",
        json={
            "init_info": {
                "buffer_id": "bench-weights",
                "buffer_size_bytes": buffer_size,
            }
        },
        timeout=120,
    )
    resp.raise_for_status()

    ctx = WPIWeightTransferEngine.trainer_init(
        dict(buffer_id="bench-weights", buffer_size_bytes=buffer_size),
        target_node_ids=target_nodes,
    )

    # ── Benchmark iterations ────────────────────────────────────────────
    pack_times = []
    propagate_times = []
    metadata_times = []
    total_times = []

    all_iters = warmup_iters + num_iters

    for i in range(all_iters):
        is_warmup = i < warmup_iters
        label = f"warmup {i + 1}/{warmup_iters}" if is_warmup else f"iter {i - warmup_iters + 1}/{num_iters}"
        logger.info("── %s ──", label)

        # Pause
        requests.post(f"{vllm_url}/pause", params={"mode": "abort"}, timeout=60).raise_for_status()

        total_start = time.perf_counter()

        # ── Pack ────────────────────────────────────────────────────────
        pack_start = time.perf_counter()
        names, dtype_names, shapes, offsets = [], [], [], []
        offset = 0
        for name, param in model.named_parameters():
            weight = param.detach().contiguous()
            nbytes = weight.nbytes
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))
            offsets.append(offset)

            ctx.vram_buffer[offset:offset + nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += nbytes

        torch.cuda.synchronize()
        pack_elapsed = time.perf_counter() - pack_start

        # ── Propagate ───────────────────────────────────────────────────
        prop_start = time.perf_counter()
        ctx.client.propagate(
            buffer_id=ctx.buffer_id,
            target_node_ids=ctx.target_node_ids,
        )
        prop_elapsed = time.perf_counter() - prop_start

        # ── Send metadata ───────────────────────────────────────────────
        meta_start = time.perf_counter()
        update_info = _asdict(WPIWeightTransferUpdateInfo(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            offsets=offsets,
            total_bytes=offset,
        ))
        resp = requests.post(
            f"{vllm_url}/update_weights",
            json={"update_info": update_info},
            timeout=300,
        )
        resp.raise_for_status()
        meta_elapsed = time.perf_counter() - meta_start

        total_elapsed = time.perf_counter() - total_start

        # Resume
        requests.post(f"{vllm_url}/resume", timeout=60).raise_for_status()

        if not is_warmup:
            pack_times.append(pack_elapsed)
            propagate_times.append(prop_elapsed)
            metadata_times.append(meta_elapsed)
            total_times.append(total_elapsed)

        bw = (total_bytes / 1024**3) / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            "  pack=%.1fms  propagate=%.1fms  metadata=%.1fms  total=%.1fms  bw=%.2f GB/s",
            pack_elapsed * 1000,
            prop_elapsed * 1000,
            meta_elapsed * 1000,
            total_elapsed * 1000,
            bw,
        )

    return BenchmarkResult(
        model_name=model_name,
        num_params=num_params,
        total_bytes=total_bytes,
        num_iters=num_iters,
        pack_times=pack_times,
        propagate_times=propagate_times,
        metadata_times=metadata_times,
        total_times=total_times,
    )


def main():
    parser = argparse.ArgumentParser(description="WPI + vLLM weight update benchmark")
    parser.add_argument("--vllm-url", required=True, help="vLLM server URL")
    parser.add_argument("--target-nodes", required=True, help="Comma-separated node IPs")
    parser.add_argument("--model", default="meta-llama/Llama-3-8B", help="Model name")
    parser.add_argument("--buffer-size-gb", type=float, default=0, help="Buffer GiB (0=auto)")
    parser.add_argument("--num-iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--output", default="", help="Path to save JSON results")
    args = parser.parse_args()

    target_nodes = [n.strip() for n in args.target_nodes.split(",")]

    result = run_benchmark(
        args.vllm_url,
        target_nodes,
        args.model,
        args.buffer_size_gb,
        args.num_iters,
        args.warmup_iters,
    )

    summary = result.summary()

    print("\n" + "═" * 70)
    print("WPI + vLLM Weight Update Benchmark Results")
    print("═" * 70)
    print(json.dumps(summary, indent=2))
    print("═" * 70)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
