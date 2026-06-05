# Weight Propagation Interface (WPI) Detailed Design Document

## 1. Introduction

The **Weight Propagation Interface (WPI)** is a Kubernetes-native orchestration framework designed to enable high-speed, zero-copy movement of large ML model weights between AI accelerators (GPUs, TPUs) across nodes in a cluster. 

As models grow to hundreds of billions of parameters, the traditional path of saving weights to shared storage (like NFS or GCS) and having each pod independently download and load them into GPU RAM becomes a severe bottleneck. WPI solves this by treating Model Weights as first-class scheduling and hardware resources, leveraging native hardware interconnects (like NVLink and InfiniBand via NCCL) and Dynamic Resource Allocation (DRA) patterns to securely and efficiently distribute weights directly into accelerator memory.

## 2. Architecture Overview

WPI consists of three main architectural layers:
1.  **The API / CRDs (The Abstraction):** Kubernetes Custom Resources that define logical blocks of weights and how workloads bind to them.
2.  **The WPI Operator (The Brain):** A Kubernetes controller that reconciles the desired distribution of weights with the cluster's physical topology.
3.  **The WPI Driver / Node Agent (The Mover):** A privileged daemonset running on accelerator nodes that executes hardware-specific commands (CUDA IPC, NCCL, TPUNode) to allocate, share, and transmit memory.
4.  **The Consumer (The ML Workload):** The ML framework (e.g., PyTorch, vLLM) that natively binds to the shared weight memory without allocating a duplicate copy.

---

## 3. Component Deep Dive

### 3.1 Custom Resource Definitions (CRDs)

WPI introduces two core custom resources acting as the Control Plane Interface:

*   **`WeightBuffer` (Cluster level):** Represents a global, logical reservation of model weights.
    *   **Spec fields:**
        *   `provider`: Instructs the backend on who is managing the allocation (e.g., `wpi-driver`).
        *   `size`: The total byte size requested.
        *   `sourcePath`: The location from which to initially stage the data (e.g., a path to Safetensors).
        *   `layout`: Describes the tensor layout.
        *   `retentionPolicy`: Determines if the weights persist beyond the lifecycle of requesting pods.
        *   `sharding` *(optional)*: Enables automatic model sharding. Contains:
            *   `strategy`: One of `TensorParallel`, `ExpertParallel`, `PipelineParallel`, or `Custom`.
            *   `numShards`: Number of shards to split the model into.
            *   `shardFiles` *(optional)*: Explicit list of `{index, path, sizeBytes}` for pre-split models.
            *   `filePattern` *(optional)*: Pattern for auto-discovering shard files (e.g., `model-{index:05d}-of-{total:05d}.safetensors`).
    *   **Status fields (populated by operator):**
        *   `totalShards`: Discovered number of shards.
        *   `discoveredShards`: List of `{index, path, sizeBytes, offsetBytes}` resolved from the sharding spec.

*   **`WeightClaim` (Namespace level):** Represents a specific pod or job's request to use a `WeightBuffer`. This mirrors the PersistentVolumeClaim (PVC) or DRA resource claim pattern.
    *   **Spec fields:**
        *   `sourceBuffer`: A reference to the underlying `WeightBuffer`.
        *   `propagationPolicy`: Defines caching or locality requirements.
        *   `targetLayout`: Can request reshaping of the tensor layout for the specific consumer.
        *   `shardIndex` *(optional)*: Which shard this claim requests. If omitted, the operator auto-assigns based on pod annotations (`wpi.io/shard-index`, `batch.kubernetes.io/job-completion-index`, or `ray.io/rank`).
    *   **Status fields (populated by operator):**
        *   `assignedShardIndex`: The resolved shard index for this claim.

### 3.2 WPI Operator

The WPI Operator watches for `WeightBuffer` and `WeightClaim` objects. When a pod is scheduled that references a `WeightClaim`, the Operator initiates the provisioning process. It makes gRPC calls to the appropriate node's WPI Driver to establish the memory and authorize connections.

### 3.3 WPI Driver (Node Agent)

The driver runs as a `DaemonSet` (typically in the `wpi-system` namespace) on nodes with accelerators. It implements three primary gRPC services defined in `wpi.proto`:

#### A. `IdentityService`
Used by the operator to discover the driver's capabilities (e.g., whether it supports `ON_THE_FLY_RESHAPING` or `CROSS_VENDOR_TRANSFER`).

#### B. `ControllerService`
Handles cluster-level abstractions. 
*   `CreateWeightBuffer`: Instructs the system to track a new buffer request.
*   `AuthorizeTransfer`: Validates that a requested weight transfer mapping is authorized.
*   `QueryTopology`: Retrieves accelerator interconnect topology (e.g., NVLink hierarchies) to optimize scheduling.

#### C. `NodeService`
The core data-plane interface responsible for moving and exposing memory.
*   **`NodeStageWeight`**: 
    1. Interacts directly with driver APIs (e.g., `libcuda`).
    2. Modifies memory access (`cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemAddressReserve`, `cuMemMap`).
    3. If a `source_path` is provided, parses the weights (e.g., Safetensors) and initiates a high-bandwidth zero-copy `HostToDevice` copy directly into the VRAM block.
    4. Starts a background thread hosting a UNIX socket (e.g., `/run/wpi/sockets/<buffer_id>.sock`). This socket uses `SCM_RIGHTS` `sendmsg` to pass the raw OS File Descriptor (FD) to a connecting consumer.
*   **`NodePropagate`**: 
    Initiates multi-node weight transfer. Supports two modes:
    *   **BROADCAST (default):** Rank 0 (the source) uses `ncclBcast` to push the full VRAM buffer over the high-speed network (InfiniBand/RoCE) to all target nodes simultaneously. All targets receive identical data.
    *   **SCATTER:** Each target receives a different shard of the buffer. The source uses `ncclSend`/`ncclRecv` group operations to send specific byte ranges (`offset_bytes:offset_bytes+length_bytes`) to specific targets. This enables distributing a single large model across multiple nodes with each receiving only its assigned shard.
    
    The propagation mode and shard assignments are communicated to targets via the TCP handshake that precedes NCCL operations. The driver also sends `PRE_UPDATE` notifications to local consumers before overwriting buffer contents, allowing them to flush caches (e.g., KV cache).
*   **`NodeRegisterWeight`**: 
    Returns the internal DMA-buf or shareable handle ID.
*   **`NodeTranslateAndMap`**: 
    Returns the device path (UNIX Socket path) to be mounted into the consumer pod.

### 3.4 The Consumer (PyTorch/ML Framework)

The consumer is standard ML framework code deployed as a pod, with a few modifications to support WPI memory mapping.

1.  **Initial Connection:** The pod waits for the WPI UNIX socket to appear in its mounted directory (`/run/wpi/sockets`).
2.  **FD Reception:** It connects to the socket and receives the shared File Descriptor via ancillary data (`recvmsg`).
3.  **CUDA Import:** Using `libcuda` via ctypes, the consumer calls `cuMemImportFromShareableHandle(fd)`, reserves an address space (`cuMemAddressReserve`), and maps the memory (`cuMemMap`).
4.  **Zero-Copy Wrapping:** Using PyTorch's `__cuda_array_interface__`, the raw physical GPU pointer is exposed as a native PyTorch Tensor. 
    ```python
    class RawCUDATensor:
        def __init__(self, ptr, nbytes):
            self.__cuda_array_interface__ = {
                "shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3
            }
    tensor = torch.as_tensor(RawCUDATensor(device_ptr, size), device='cuda:0')
    weights = tensor.view(torch.float16)
    ```
This bypasses memory allocation inside the pod entirely. The weights exist in the memory allocated by the WPI driver, and the PyTorch process merely holds a reference to that physical memory.

---

## 4. End-to-End Workflow (VRAM Mapping)

1.  **Cluster Admin:** Creates a `WeightBuffer` pointing to an NFS directory of Safetensors.
2.  **Operator:** Sees the `WeightBuffer` and waits for consumers.
3.  **ML Job:** Creates a Pod with a `WeightClaim` for that buffer.
4.  **Operator/Scheduler:** Schedules the Pod to `Node A`. Calls `NodeStageWeight` on `Node A`'s driver.
5.  **Driver on Node A:**
    *   Allocates contiguous VRAM.
    *   Reads the Safetensors from NFS directly into the VRAM chunk.
    *   Exports the memory as an FD and creates an `SCM_RIGHTS` UNIX socket.
6.  **Pod on Node A:** Starts up, queries the WPI socket, gets the FD, maps it using `cuMemImportFromShareableHandle`, wraps the pointer in a framework tensor, and begins inference instantly.

---

## 5. End-to-End Workflow (Multi-Node Propagation)

1.  **ML Distributed Job:** A massive inference job is scheduled across Nodes A, B, C, and D.
2.  **Operator:** Detects that the model weights are currently only staged on `Node A`. It pre-allocates uninitialized memory on Nodes B, C, and D by calling `NodeStageWeight` (without a source path).
3.  **Operator:** Calls `NodePropagate` on `Node A`'s WPI driver, providing the IPs of nodes B, C, and D as targets.
4.  **Drivers:** Node A spawns an NCCL Unique ID. Nodes B, C, and D connect via TCP to get the ID and their collective Rank.
5.  **NCCL Operation:** Node A calls `ncclBcast()`. The weights fly over the RDMA backend at ~200+ Gbps switch fabric speeds simultaneously into the VRAM of nodes B, C, and D.
6.  **Pods:** All pods across all 4 nodes map the memory locally and begin distributed inference.

---

## 6. Future Capabilities & Extensibility

*   **P2P Topology Awareness:** Integrating with PCIe/NVLink topology APIs to perfectly schedule readers based on hardware distance (NUMA node, NVLink bridge).
*   **TPU Support:** Expanding the memory abstraction from `libcuda` and `cuMem` over to `libtpu` equivalent memory export constructs.
*   **On-the-Fly Reshaping:** Implementing custom CUDA kernels in the WPI driver to transpose or shard memory blocks (Column-major to Row-major) dynamically as it is passed to the consumer, accommodating different framework needs from the same physical copy.

---

## 7. Sharding Support

WPI supports first-class model sharding, enabling a single `WeightBuffer` to be automatically distributed across multiple GPUs and nodes. This is critical for models that exceed single-GPU memory (e.g., Kimi K2 at 1T parameters, Llama 405B).

### 7.1 Sharding Strategies

| Strategy | Description | Use Case |
|---|---|---|
| `TensorParallel` | Even byte-range splits across GPUs | Most common; each GPU gets a contiguous slice |
| `ExpertParallel` | Expert-to-GPU mapping for MoE models | Kimi K2 (384 experts), Mixtral |
| `PipelineParallel` | Layer blocks assigned to different stages | Deep models split by layer groups |
| `Custom` | User-defined shard-to-file mapping | Pre-sharded checkpoints |

### 7.2 Shard Discovery

The WPI operator resolves shards via three paths (in priority order):
1. **Explicit `shardFiles`** — user provides exact `{index, path, sizeBytes}` per shard
2. **`filePattern`** — operator constructs paths like `model-00001-of-00008.safetensors`
3. **Capacity split** — operator divides `size / numShards` into even byte ranges with `offsetBytes`

### 7.3 Shard-Scoped Buffers

The driver tracks sharded buffers using a naming convention: `<buffer_id>__shard_<N>`. Each shard gets its own:
- VRAM allocation
- FD-passing UNIX socket (`<buffer_id>__shard_<N>.sock`)
- Notification socket (`<buffer_id>__shard_<N>_notify.sock`)

This is transparent to the consumer — the `WPIClient` resolves the correct shard-scoped ID automatically.

### 7.4 End-to-End Sharded Propagation

1. **Admin:** Creates a `WeightBuffer` with `sharding: {strategy: TensorParallel, numShards: 8}`.
2. **Operator:** Discovers 8 shards, populates `status.discoveredShards` with paths/sizes.
3. **Job:** Creates 8 `WeightClaim` objects. Shard indices are auto-assigned from pod annotations.
4. **Driver (source):** Stages the full model into a single VRAM buffer.
5. **Propagation:** The operator calls `NodePropagate` with `mode: SCATTER` and `shard_assignments` mapping each target to its byte range.
6. **Driver (targets):** Each target receives only its shard via `ncclRecv`, deposits it at the correct offset in its local buffer.
7. **Pods:** Each pod maps its local shard and begins parallel execution.

---

## 8. Benchmark Results

### 8.1 Environment

| Component | Specification |
|---|---|
| **Cluster** | GKE (us-central1), 2× A4 nodes |
| **GPUs** | 8× NVIDIA GPUs per node (A100-equivalent) |
| **Interconnect** | InfiniBand with RDMA/GDR (GPUDirect RDMA) |
| **NCCL** | With IB plugin, adaptive routing enabled |
| **Kubernetes** | DRA (Dynamic Resource Allocation) for GPU memory scheduling |

### 8.2 8-Shard Scatter Propagation (600 GB)

Full 8-GPU concurrent transfer: each of the 8 source GPUs sends its 75 GB shard to the corresponding target GPU on the remote node over a dedicated NCCL communicator. All 8 streams run simultaneously.

**Configuration:**
- `WeightBuffer`: 600 GiB, `TensorParallel`, `numShards: 8`
- 16 DRA pods total (8 source + 8 target)
- 8 independent NCCL communicators (world_size=2 each)

**Results (best of 3 runs):**

| Metric | Value |
|---|---|
| Total Data Transferred | 600 GB |
| Number of Concurrent Streams | 8 |
| Shard Size | 75 GB |
| Average Stream Latency | 2.31 s |
| Max Stream Latency (wall clock) | 2.39 s |
| Per-Stream Bandwidth (avg) | 32.43 GB/s |
| **Aggregate Cross-Node Throughput** | **251 GB/s** |

**Per-Shard Breakdown:**

| Shard | Latency (s) | Bandwidth (GB/s) |
|---|---|---|
| 0 | 2.2654 | 33.11 |
| 1 | 2.2887 | 32.77 |
| 2 | 2.2536 | 33.28 |
| 3 | 2.3122 | 32.44 |
| 4 | 2.3310 | 32.17 |
| 5 | 2.3901 | 31.38 |
| 6 | 2.3543 | 31.86 |
| 7 | 2.3034 | 32.56 |

### 8.3 Key Observations

- **Near-linear scaling:** 8 concurrent NCCL streams achieve ~251 GB/s aggregate, close to the theoretical 8× single-stream bandwidth. The IB fabric distributes traffic evenly across NIC ports.
- **Low variance:** The spread between fastest (shard 2: 2.25s) and slowest (shard 5: 2.39s) is only ~140ms, indicating minimal contention.
- **NCCL transport:** All streams use `NET/IB/GDRDMA/Shared` — zero-copy GPU-to-GPU over InfiniBand with GPUDirect RDMA, bypassing host memory entirely.
- **Concurrent bootstrap:** Each shard's NCCL communicator initializes independently in parallel. No serialization locks are needed; the `/dev/shm` volume is sized to 1 GiB to accommodate the shared memory segments required by 8 concurrent NCCL proxy threads.

---

## 9. Sharded Weight Propagation Architecture

To support large-scale distributed workloads (e.g., Tensor Parallelism, Expert Parallelism), WPI implements a highly flexible, decoupled sharded weight propagation architecture.

### 9.1 Decoupled GPU Topology (1-to-N Scatter)

The WPI sharding architecture is designed to decouple the publisher and consumer topologies. Under the hood, WPI operates as a **1-to-N Scatter** model where a single publisher (Trainer Rank 0) acts as the coordinator holding the full, centralized model weights and scatters them directly to N consumer GPUs (e.g. SGLang TP ranks).

This design completely avoids requiring matching node-level GPU densities or symmetric cluster-wide layouts between the trainer and inference groups.

```
  1-TO-N SCATTER TOPOLOGY (VERIFIED IN PRODUCTION):

  [ Trainer Node A (GPU 0) ]  <--- (Single Publisher holding 8GB full model)
             │
             ├───(4GB Shard 0)───> [ Inference Node B (GPU 0 - TP0) ]
             └───(4GB Shard 1)───> [ Inference Node B (GPU 1 - TP1) ]
```

*   **Publisher Side**: The Trainer Rank 0 stages a single, global weight buffer representing the full model weights (e.g., 8GB). It does not need to know SGLang's internal GPU layout during staging.
*   **Consumer Side**: Each consumer worker stages and maps only its respective slice (e.g., SGLang TP0 stages a 4GB shard 0, TP1 stages a 4GB shard 1). This saves substantial VRAM on inference nodes, as they only allocate memory for the partition they actually execute.
*   **Decoupled Mapping**: During `propagate()`, Trainer Rank 0 defines a list of `ShardAssignment`s mapping byte offsets in its global buffer to target node IPs, shard indexes, and target GPU IDs. 

**Verification Status**: We have successfully validated this **1-to-2 Scatter** layout in our active training run:
*   **Sender (Publisher)**: A single GPU (GPU 0) on Trainer Node 1 (`10.60.2.63`) holding the full 8GB weight buffer.
*   **Receivers (Consumers)**: Two separate GPUs (GPU 0 and GPU 1) on SGLang Serving Node (`10.60.3.77`).

Trainer Rank 0's WPI driver successfully initialized a 3-rank cross-node NCCL communicator group and scattered the two 4GB shards concurrently over GKE's network to SGLang's separate GPUs. SGLang TP1 correctly received its shard on GPU 1 without requiring a symmetric 2-to-2 GPU layout.

---

### 9.2 Shard-Scoped Buffer Management

The WPI driver manages sharded buffers by dynamically appending a suffix to the base buffer name:
```
<buffer_id>__shard_<shard_index>
```
For example, if the base buffer is `slime-weights`, the driver creates `slime-weights__shard_0` and `slime-weights__shard_1`.

*   **Isolation**: Each shard has its own dedicated UNIX domain socket (e.g., `/run/wpi/sockets/slime-weights__shard_0.sock`) and notification socket.
*   **Transparent Suffixing**: The Python `WPIClient` handles this transparently. When a consumer calls `receive_fd(buffer_id="slime-weights", shard_index=1, total_shards=2)`, the client automatically connects to the correct shard-scoped socket, ensuring workers only receive the file descriptor for their assigned slice.

---

### 9.3 Dynamic NCCL Communicator Grouping

To perform the high-speed cross-node transfer, the WPI driver dynamically orchestrates a single cross-node NCCL communicator group at runtime:

1.  **Topology Discovery**: The publisher's `propagate()` request contains the target IPs and GPU IDs. The driver computes the total communicator size: `1 (publisher) + total_unique_target_gpus`.
2.  **Point-to-Point Streaming**: Instead of broadcasting the entire buffer to all nodes, the driver executes parallel point-to-point `ncclSend` (on the publisher node) and `ncclRecv` (on the receiver nodes) operations.
3.  **GPUDirect RDMA**: The data streams directly from the publisher's VRAM over the network fabric (InfiniBand/RoCE/TCP) into the receiver's mapped VRAM shard buffer, bypassing host memory and CPU serialization entirely.

---

### 9.4 Deadlock Prevention via Explicit GPU Routing

In multi-GPU/multi-rank container environments (such as SGLang TP=2), multiple consumer ranks share the same host directory `/run/wpi/sockets`. 

To prevent deadlocks and VRAM relocation overhead, the SCM_RIGHTS file-descriptor passing protocol implements **Explicit GPU Routing**:

1.  **GPU Metadata Exchange**: When a consumer rank connects to the driver's UNIX socket to retrieve its VRAM file descriptor, it first sends a metadata string specifying its target GPU ID:
    ```
    GPU=<gpu_id>\n
    ```
2.  **Zero-Relocation FD Mapping**: The driver reads this metadata. If SGLang TP1 requests `GPU 1`, the driver immediately binds the FD to the context of GPU 1.
3.  **Deadlock Prevention**: This ensures the driver never attempts to perform a GPU-to-GPU VRAM relocation (e.g., trying to relocate a buffer staged on GPU 1 to GPU 0), which would block the driver's single-threaded socket listener and deadlock the other TP ranks.


