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

"""Unit tests for the vLLM WPI WeightTransferEngine.

Tests the WPI engine's dataclass validation, packing/unpacking logic, and
factory registration. These tests do NOT require a running WPI driver —
they mock the WPIClient to test the engine logic in isolation.

Run:
    pytest tests/test_wpi_engine.py -v
"""

import math
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestWPIWeightTransferUpdateInfo:
    """Tests for WPIWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferUpdateInfo,
        )

        info = WPIWeightTransferUpdateInfo(
            names=["layer.0.weight", "layer.0.bias"],
            dtype_names=["bfloat16", "bfloat16"],
            shapes=[[768, 768], [768]],
            offsets=[0, 768 * 768 * 2],
            total_bytes=768 * 768 * 2 + 768 * 2,
        )
        assert len(info.names) == 2
        assert info.total_bytes == 768 * 768 * 2 + 768 * 2

    def test_mismatched_names_dtypes_raises(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferUpdateInfo,
        )

        with pytest.raises(ValueError, match="dtype_names.*same size"):
            WPIWeightTransferUpdateInfo(
                names=["a", "b"],
                dtype_names=["bfloat16"],  # wrong length
                shapes=[[10], [10]],
                offsets=[0, 20],
            )

    def test_mismatched_shapes_raises(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferUpdateInfo,
        )

        with pytest.raises(ValueError, match="shapes.*same size"):
            WPIWeightTransferUpdateInfo(
                names=["a"],
                dtype_names=["float32"],
                shapes=[[10], [20]],  # wrong length
                offsets=[0],
            )

    def test_mismatched_offsets_raises(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferUpdateInfo,
        )

        with pytest.raises(ValueError, match="offsets.*same size"):
            WPIWeightTransferUpdateInfo(
                names=["a", "b"],
                dtype_names=["float32", "float32"],
                shapes=[[10], [10]],
                offsets=[0],  # wrong length
            )

    def test_serialization_roundtrip(self):
        """Ensure update_info survives dict → dataclass conversion (HTTP path)."""
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferUpdateInfo,
        )

        original = WPIWeightTransferUpdateInfo(
            names=["w1", "w2"],
            dtype_names=["bfloat16", "float32"],
            shapes=[[128, 64], [64]],
            offsets=[0, 128 * 64 * 2],
            total_bytes=128 * 64 * 2 + 64 * 4,
        )
        d = asdict(original)
        restored = WPIWeightTransferUpdateInfo(**d)
        assert restored.names == original.names
        assert restored.offsets == original.offsets
        assert restored.total_bytes == original.total_bytes


class TestWPIWeightTransferInitInfo:
    """Tests for WPIWeightTransferInitInfo defaults and validation."""

    def test_defaults(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferInitInfo,
        )

        info = WPIWeightTransferInitInfo()
        assert info.buffer_id == "vllm-weights"
        assert info.socket_dir == "/run/wpi/sockets"
        assert info.shard_index == -1
        assert info.total_shards == 0

    def test_custom_values(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferInitInfo,
        )

        info = WPIWeightTransferInitInfo(
            buffer_id="my-model",
            buffer_size_bytes=10 * 1024**3,
            shard_index=2,
            total_shards=8,
        )
        assert info.buffer_id == "my-model"
        assert info.shard_index == 2

    def test_dict_roundtrip(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferInitInfo,
        )

        d = {"buffer_id": "test", "buffer_size_bytes": 1024}
        info = WPIWeightTransferInitInfo(**d)
        assert info.buffer_id == "test"
        assert info.buffer_size_bytes == 1024


class TestWPITrainerSendWeightsArgs:
    """Tests for WPITrainerSendWeightsArgs validation."""

    def test_http_mode_requires_url(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPITrainerSendWeightsArgs,
        )

        with pytest.raises(ValueError, match="url is required"):
            WPITrainerSendWeightsArgs(mode="http")

    def test_ray_mode_requires_handle(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPITrainerSendWeightsArgs,
        )

        with pytest.raises(ValueError, match="llm_handle is required"):
            WPITrainerSendWeightsArgs(mode="ray")

    def test_invalid_mode_raises(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPITrainerSendWeightsArgs,
        )

        with pytest.raises(ValueError, match="mode must be"):
            WPITrainerSendWeightsArgs(mode="grpc")

    def test_http_mode_valid(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPITrainerSendWeightsArgs,
        )

        args = WPITrainerSendWeightsArgs(
            mode="http", url="http://localhost:8000"
        )
        assert args.url == "http://localhost:8000"


class TestWPIEngineReceiveWeights:
    """Tests for receive_weights() unpacking logic with mocked WPIClient."""

    def test_receive_unpacks_correctly(self):
        """Verify tensors are correctly sliced from the flat VRAM buffer."""
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferEngine,
            WPIWeightTransferUpdateInfo,
        )

        # Create a fake engine with a pre-populated VRAM buffer
        config = MagicMock()
        parallel_config = MagicMock()
        engine = WPIWeightTransferEngine(config, parallel_config)

        # Simulate two weights packed into a flat buffer
        w1 = torch.randn(4, 8, dtype=torch.bfloat16, device="cpu")
        w2 = torch.randn(16, dtype=torch.float32, device="cpu")

        buf = torch.zeros(w1.nbytes + w2.nbytes, dtype=torch.uint8, device="cpu")
        buf[:w1.nbytes].copy_(w1.view(-1).view(torch.uint8))
        buf[w1.nbytes:w1.nbytes + w2.nbytes].copy_(w2.view(-1).view(torch.uint8))

        engine._vram_buffer = buf
        engine._client = MagicMock()
        engine._client.wait_for_ready = MagicMock()

        update_info = WPIWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["bfloat16", "float32"],
            shapes=[[4, 8], [16]],
            offsets=[0, w1.nbytes],
            total_bytes=w1.nbytes + w2.nbytes,
        )

        received = []

        def mock_load_weights(weights):
            received.extend(weights)

        engine.receive_weights(update_info, mock_load_weights)

        # Verify
        assert len(received) == 2
        assert received[0][0] == "layer.weight"
        assert received[1][0] == "layer.bias"
        assert torch.equal(received[0][1], w1)
        assert torch.equal(received[1][1], w2)
        engine._client.wait_for_ready.assert_called_once()

    def test_receive_not_initialized_raises(self):
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferEngine,
            WPIWeightTransferUpdateInfo,
        )

        config = MagicMock()
        parallel_config = MagicMock()
        engine = WPIWeightTransferEngine(config, parallel_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            engine.receive_weights(
                WPIWeightTransferUpdateInfo(
                    names=[], dtype_names=[], shapes=[], offsets=[]
                ),
                lambda w: None,
            )


class TestWPIEngineFactoryRegistration:
    """Tests that the WPI engine is properly registered in the factory."""

    def test_wpi_registered(self):
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        assert "wpi" in WeightTransferEngineFactory._registry

    def test_wpi_creates_engine(self):
        """Test lazy loading creates the right engine type."""
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory
        from vllm.distributed.weight_transfer.wpi_engine import (
            WPIWeightTransferEngine,
        )

        config = MagicMock()
        config.backend = "wpi"
        parallel_config = MagicMock()

        engine = WeightTransferEngineFactory.create_engine(config, parallel_config)
        assert isinstance(engine, WPIWeightTransferEngine)
