# Weight Propagation Interface (WPI) - End-to-End User Guide

This guide provides step-by-step instructions on how to use the Weight Propagation Interface (WPI) to orchestrate model weight distribution and perform zero-copy VRAM mapping for ML inference workloads on Kubernetes.

## Prerequisites

Before using WPI, ensure your environment meets the following requirements:
1. **Kubernetes Cluster**: A GKE (or other Kubernetes) cluster with GPU-enabled nodes (e.g., L4, A100, H100).
2. **Dynamic Resource Allocation (DRA)**: The cluster must support DRA (usually enabled via feature gates in K8s 1.30+).
3. **WPI Control Plane**: The WPI Operator and Custom Resource Definitions (`WeightBuffer`, `WeightClaim`) must be installed.
4. **WPI Data Plane**: The `wpi-driver` DaemonSet must be running on your GPU nodes and must have privileged access to map memory and manage UNIX sockets.
5. **Shared Storage**: A shared filesystem (e.g., Filestore / NFS / GCSFuse) mounted on the GPU nodes for staging initial weights.

---

## Step 1: Prepare Your Model Weights

The WPI system loads weights directly from shared storage into GPU Memory. For optimal performance, weights should be stored in the **Safetensors** format.

1. Download or convert your model to `.safetensors` format.
2. Place the model files on your shared network storage.
3. Ensure the storage is accessible at a specific path on all GPU nodes (e.g., `/mnt/nfs/models/llama-3-8b/model.safetensors`).

---

## Step 2: Provision a WeightBuffer (Platform Admin)

A `WeightBuffer` is a cluster-level resource that tells WPI to reserve a block of GPU memory and load a specific set of weights into it.

Create a file named `weightbuffer.yaml`:

```yaml
apiVersion: wpi.sig.k8s.io/v1alpha1
kind: WeightBuffer
metadata:
  name: llama-3-8b-weights
spec:
  provider: wpi-driver
  size: "16Gi" # The total size of the required VRAM
  sourcePath: "/mnt/nfs/models/llama-3-8b/model.safetensors"
  layout: ROW_MAJOR
  retentionPolicy: Retain # Keep memory allocated even if no pods are using it
```

Apply it to the cluster:
```bash
kubectl apply -f weightbuffer.yaml
kubectl get weightbuffers
```
*Behind the scenes*: The WPI Operator tracks this resource but doesn't allocate physical memory until a consumer requests it, or unless pre-staging is explicitly triggered.

---

## Step 3: Claim the Weights (Data Scientist)

A `WeightClaim` is a namespace-scoped resource similar to a PersistentVolumeClaim (PVC). It requests access to a specific `WeightBuffer`.

Create a file named `weightclaim.yaml`:

```yaml
apiVersion: wpi.sig.k8s.io/v1alpha1
kind: WeightClaim
metadata:
  name: my-llama-claim
  namespace: default
spec:
  sourceBuffer: llama-3-8b-weights
  propagationPolicy: LocalHost # Can be Remote, LocalHost, etc.
```

Apply it to the cluster:
```bash
kubectl apply -f weightclaim.yaml
kubectl get weightclaims
```

---

## Step 4: Consume Weights in Your ML Workload

Your ML workload (a Pod running PyTorch, vLLM, etc.) will reference the `WeightClaim` using standard Kubernetes DRA syntax.

### 4.1 Define the Pod

Create `ml-job.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llama-inference-node
  namespace: default
spec:
  containers:
  - name: inference-container
    image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
    command: ["python", "/app/consumer.py"]
    volumeMounts:
    # Mount the standard WPI socket directory
    - name: wpi-sockets
      mountPath: /run/wpi/sockets
  volumes:
  - name: wpi-sockets
    hostPath:
      path: /run/wpi/sockets
      type: DirectoryOrCreate
  # Use DRA to request the WeightClaim
  resourceClaims:
  - name: wpi-weights
    source:
      resourceClaimName: my-llama-claim
```

*Behind the scenes*: When you apply this pod, the Kubernetes scheduler and WPI Operator negotiate. The WPI Driver on the scheduled node uses `cuMemCreate` + `cuMemExportToShareableHandle` to allocate the VRAM, loads the Safetensors data into it via zero-copy mapping, and opens a UNIX socket.

### 4.2 Application Code (PyTorch Example)

Inside `/app/consumer.py`, your application must connect to the WPI local socket, receive the File Descriptor, and wrap it into a PyTorch Tensor. WPI provides standard patterns for this.

```python
import socket
import ctypes
import array
import torch
import os

# 1. Wait for and connect to WPI Driver UNIX Socket
sock_path = "/run/wpi/sockets/llama-3-8b-weights.sock"
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(sock_path)

# 2. Receive the File Descriptor via SCM_RIGHTS
fds = array.array("i", [0])
msg, ancdata, flags, addr = client.recvmsg(1, socket.CMSG_LEN(fds.itemsize))
for cmsg_level, cmsg_type, cmsg_data in ancdata:
    if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
        fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
        fd = fds[1]

# 3. Import the Shareable Handle using libcuda
libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
libcuda.cuInit(0)

# Retrieve generic handle and map to reserved device address space
# (Refer to WPI Driver design for specific ctypes mapping logic)
# device_ptr = mapped_vram_address
# size = 16 * 1024 * 1024 * 1024 # 16 GiB

# 4. Expose the raw VRAM safely to PyTorch via __cuda_array_interface__
class WPITensor:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

raw_array = WPITensor(device_ptr, size)
tensor = torch.as_tensor(raw_array, device='cuda:0')
weights = tensor.view(torch.float16)

print("Zero-Copy weights loaded successfully! Shape:", weights.shape)
print("Starting inference...")
# The process exits without freeing the VRAM, as WPI owns it!
```

Deploy your ML Job:
```bash
kubectl apply -f ml-job.yaml
```

---

## Step 5: Multi-Node Broadcast (Scaling Out)

For Massive distributed ML algorithms requiring pipeline or tensor parallelism, WPI can broadcast weights across native interconnects.

When a distributed Job is launched (using MPIJob, RayJob, or JobSet), the WPI operator detects multiple nodes need the same `WeightBuffer`.

1. The WPI Operator triggers `NodePropagate` on the root node (where weights are currently staged).
2. The Driver spins up an `NcclCommunicator` connecting to the remote drivers on the target nodes.
3. Node 0 broadcasts the weights via `cupy.cuda.nccl.bcast()`, utilizing InfiniBand/RoCE.
4. All remote nodes receive the weights *directly* into their pre-allocated VRAM buffers at 200+ Gbps fabric speed.
5. All ML distributed pods subsequently map their local VRAM copies simultaneously and begin execution.

This is all completely abstracted from the end-user. The only requirement is that all distributed pods reference the exact same `WeightClaim` in their pod spec, and the WPI Operator handles the background propagation perfectly!

---

## Benchmark Results

### Intra-Node Performance
Testing on a single a4-highcpu-8g instance demonstrated a sustained bandwidth of ~650 GB/s. A 20GB buffer was transferred between Rank 0 (GPU 0) and Rank 1 (GPU 1) via NVLink in just 0.03 seconds.

### Cross-Node Sockets (TCP) Results
Utilizing two g2-standard-4 machines, a 10GB tensor was mapped and transmitted across isolated Pod network interfaces using standard unoptimized TCP networking.
- **Performance:** Maintained a steady baseline transfer rate of **1.00 GB/s**.
- **Latency:** The process took **~20.00 seconds** to complete, demonstrating the high CPU overhead and latency inherent to standard socket transfers for large tensors.
- **Speed of Light Context:** The g2-standard-4 maximum network bandwidth is 1.25 GB/s. The transfer achieved **80% of maximum bandwidth**.

### Distributed Broadcast Validation (1-to-N)
Utilizing three g2-standard-4 machines, a 10GB tensor was mapped and transmitted across isolated Pod network interfaces using standard unoptimized TCP networking.
- **Architectural Setup:** The GKE l4-pool was scaled to three identical physical instances to support refactored testing protocols.
- **Validation Outcome:** A 10GB payload (comprising 7.0 fp16 scalars) was broadcast natively from Rank 0 to distributed targets at Rank 1 and Rank 2. Target nodes reported concurrent NCCL completion in **11.0381s** with a bandwidth of **0.91 GB/s**. Successful data integrity tests confirm that WPI can effectively bootstrap a broadcast topology across three isolated Pod networks.
- **Speed of Light Context:** The g2-standard-4 maximum network bandwidth is 1.25 GB/s. The transfer achieved **72.8% of maximum bandwidth**.

### A3-Ultragpu-8g RDMA Benchmark (Baseline vs. Optimized)
Validation of zero-copy broadcast operations was conducted by transmitting a 10 GiB payload across internal Pod networks using two distinct configurations to compare baseline socket routing against hardware-accelerated routing.

**Baseline: Standard Socket-Based**
- **Payload Volume:** 10 GiB of allocated tensor memory.
- **Bandwidth:** Sustained throughput of **5.50 GB/s**.
- **Latency:** Completion time of **1.811 seconds**.

**Optimized: GPUDirect RDMA-Enabled**
- **Payload Volume:** 75 GiB.
- **Bandwidth:** Sustained throughput of **36.57 GB/s**.
- **Latency:** Minimized transfer latency of **2.0379 seconds**.

**Performance Delta & Speed of Light Analysis**
- **Throughput Increase:** RDMA acceleration delivered a **565% increase** in bandwidth over the baseline socket transfer.
- **Speed of Light Context:** The a3-ultragpu-8g maximum network bandwidth per NIC is 50 GB/s. The transfer achieved **73.14% of maximum bandwidth**. For comparison, Ray on A4 achieved 35 GB/s (70% of maximum bandwidth).
