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
        *   `shardIndex` *(optional)*: Which shard this claim requests. If omitted, the operator auto-assigns based on pod annotations (`wpi.sig.k8s.io/shard-index`, `batch.kubernetes.io/job-completion-index`, or `ray.io/rank`).
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

