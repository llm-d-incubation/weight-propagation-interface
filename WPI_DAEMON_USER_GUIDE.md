# Weight Propagation Interface (WPI) - Daemon Mode User Guide

This guide describes how to run and consume the **Weight Propagation Interface (WPI)** as a standard Node DaemonSet without relying on Kubernetes Custom Resource Definitions (CRDs) or the Dynamic Resource Allocation (DRA) framework. 

Running WPI in **Daemon Mode** allows you to orchestrate weight distribution directly from your ML scripts (e.g., PyTorch, JAX, or Ray jobs) using gRPC/UNIX socket APIs.

---

## Architecture Overview (Daemon Mode)

In Daemon Mode, the WPI Driver runs as a privileged DaemonSet on each accelerator node. Application pods communicate with the local driver daemon via a shared host directory mapping UNIX domain sockets:

```
+---------------------------------------------------------------------------------+
|                                  PHYSICAL NODE                                  |
|                                                                                 |
|  +---------------------------+              +--------------------------------+  |
|  |   Application Container   |              |       WPI Driver Daemon        |  |
|  |  (PyTorch / JAX / Ray)    |              |          (Privileged)          |  |
|  |                           |              |                                |  |
|  |  1. Calls WPIClient       |  gRPC Socket |  2. Allocates VRAM             |  |
|  |     stage_weight()  ======|==============>  3. Creates FD-passing Socket  |  |
|  |                           |  (unix://)   |                                |  |
|  |  4. receive_fd()   <======|==============/                                |  |
|  |  5. Maps memory via       |              |                                |  |
|  |     CUDA VMM or mmap      |              |                                |  |
|  +---------------------------+              +--------------------------------+  |
|               ||                                             ||                 |
+---------------||---------------------------------------------||-----------------+
                ||                                             ||
                || (Zero-Copy VRAM read/write)                 || (NCCL / Fabric Network)
                \/                                             \/
        +---------------+                              +---------------+
        |  GPU/TPU VRAM |                              | Target Node   |
        +---------------+                              +---------------+
```

---

## Step 0: Build and Package the WPI Daemon

To run the WPI driver daemon (either in Kubernetes via DaemonSet or directly on host virtual machines), you must build and package it.

### Method A: Container Image Packaging (Recommended for Kubernetes)

You can build a Docker image containing the Python gRPC server using the existing `Dockerfile` located in the `driver` directory.

1. Build the container image from the root directory of the repository:
```bash
docker build -t <YOUR_IMAGE_REGISTRY>/wpi-driver:latest -f driver/Dockerfile .
```

2. Push the image to your registry:
```bash
docker push <YOUR_IMAGE_REGISTRY>/wpi-driver:latest
```

---

### Method B: Systemd / Host-Level Deployment (Recommended for VM/Bare-Metal Pools)

If you are running workloads on bare-metal nodes or VM instances directly (outside of Kubernetes), you can package and run the daemon as a host system service.

1. **Install Host Dependencies**:
   Ensure CUDA drivers are installed on the host, then run:
   ```bash
   pip install grpcio grpcio-tools numpy cupy-cuda12x nvidia-nccl-cu12 torch safetensors
   ```

2. **Generate Protobuf Bindings**:
   ```bash
   python -m grpc_tools.protoc -I./proto --python_out=./driver --grpc_python_out=./driver ./proto/wpi.proto
   ```

3. **Configure Systemd Service**:
   Create `/etc/systemd/system/wpi-driver.service` on the host:
   ```ini
   [Unit]
   Description=Weight Propagation Interface Daemon
   After=network.target nvidia-persistenced.service

   [Service]
   Type=simple
   WorkingDirectory=/path/to/weight-propagation-interface/driver
   Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib"
   ExecStart=/usr/bin/python3 main.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

4. **Enable and Start the Service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable wpi-driver
   sudo systemctl start wpi-driver
   sudo systemctl status wpi-driver
   ```

---

## Step 1: Deploy the WPI Driver DaemonSet

Deploy the `wpi-driver` on your GKE/Kubernetes node pool. The daemon must run in `hostNetwork: true` mode to allow direct driver-to-driver NCCL communication, and it requires privileged mounts to share memory descriptors with your container runtime.

Create `wpi-driver-daemonset.yaml`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: wpi-driver
  namespace: wpi-system
  labels:
    app: wpi-driver
spec:
  selector:
    matchLabels:
      app: wpi-driver
  template:
    metadata:
      labels:
        app: wpi-driver
    spec:
      hostNetwork: true
      tolerations:
      - operator: "Exists"
      containers:
      - name: wpi-driver
        image: <YOUR_IMAGE_REGISTRY>/wpi-driver:latest
        imagePullPolicy: IfNotPresent
        command: ["python", "main.py"]
        securityContext:
          privileged: true
        ports:
        - containerPort: 50051
          name: grpc
        volumeMounts:
        # UNIX sockets for gRPC and FD-passing
        - name: wpi-sockets
          mountPath: /run/wpi/sockets
        # Shared memory mapping for CPU/TPU staging
        - name: dshm
          mountPath: /dev/shm
        # Device access (GPU / TPU)
        - name: host-dev
          mountPath: /dev
      volumes:
      - name: wpi-sockets
        hostPath:
          path: /run/wpi/sockets
          type: DirectoryOrCreate
      - name: dshm
        hostPath:
          path: /dev/shm
          type: Directory
      - name: host-dev
        hostPath:
          path: /dev
          type: Directory
```

Apply it to the cluster:
```bash
kubectl apply -f wpi-driver-daemonset.yaml
```

---

## Step 2: Configure Your Application Volume Mounts

Any application pod (e.g., Trainer, Inference Server, Ray Actor) that needs to push or pull weights must mount the same `/run/wpi/sockets` host path to talk to the local node's driver.

Add this directory hostPath volume mapping to your application Pod spec:

```yaml
spec:
  containers:
  - name: my-ml-app
    image: my-registry/my-ml-app:latest
    volumeMounts:
    - name: wpi-sockets
      mountPath: /run/wpi/sockets
  volumes:
  - name: wpi-sockets
    hostPath:
      path: /run/wpi/sockets
      type: DirectoryOrCreate
```

---

## Step 3: Consume WPI in Application Code

The Python `WPIClient` library abstracts the underlying UNIX domain socket commands, gRPC client stubs, and OS/hardware memory mappings.

### 3.1 Initializing the Client
The client automatically discovers the local gRPC server on `unix:///run/wpi/sockets/wpi-grpc.sock`.

```python
from wpi_client.client import WPIClient

# Initialize client (uses UNIX sockets by default)
client = WPIClient(socket_dir="/run/wpi/sockets")
```

---

### 3.2 Staging a Weight Buffer (Memory Allocation)
Before mapping memory or propagating weights, the buffer must be staged. This instructs the WPI driver to reserve the physical memory region (VRAM on GPU or shared memory on host).

```python
buffer_id = "llama-3-8b-weights"
size_bytes = 16 * 1024 * 1024 * 1024  # 16 GiB
claim_id = "my-active-session"

# Stage the buffer (allocates memory on GPU/Host)
client.stage_weight(
    buffer_id=buffer_id,
    size_bytes=size_bytes,
    claim_id=claim_id
)
```

---

### 3.3 Zero-Copy Memory Mapping (Consumer Side)
Once staged, the consumer maps the buffer directly into its Python process. WPI passes the file descriptor of the VRAM/shm handle over a local UNIX socket and maps it zero-copy into a PyTorch tensor.

```python
# 1. Retrieve the file descriptor via SCM_RIGHTS unix domain socket passing
fd = client.receive_fd(buffer_id=buffer_id, gpu_id=0)

# 2. Map the file descriptor directly into the CUDA Virtual Memory space (GPU VMM)
device_ptr = client.import_cuda_memory(fd=fd, size_bytes=size_bytes, device_id=0)

# 3. Expose raw memory as a zero-copy PyTorch tensor via __cuda_array_interface__
raw_tensor = client.wrap_as_buffer(device_ptr=device_ptr, size_bytes=size_bytes)

# 4. View raw bytes as the model's datatype
model_weights = raw_tensor.view(torch.float16)

print("Mapped zero-copy GPU tensor. Address:", hex(device_ptr))
print("Tensor view capacity:", model_weights.shape)
```

---

### 3.4 Multi-Host Weight Propagation (Trainer / Publisher Side)
The trainer node populates its local mapped memory with updated weights, then triggers multi-host propagation to copy them to serving nodes using fast networking (NCCL or custom TCP sockets).

#### Broadcast Mode (1-to-N)
Sends the entire weight buffer to all serving nodes:

```python
# target_node_ids must contain target node IPs reachable over hostNetwork
target_ips = ["10.130.0.85", "10.130.0.86"]

# Broadcast full buffer to all target hosts
client.propagate(
    buffer_id=buffer_id,
    target_node_ids=target_ips,
    mode=0  # 0 = BROADCAST
)
```

#### Scatter Mode (TP / Partitioned)
Sends distinct slices of the local buffer to different target nodes (useful for Tensor Parallel model shard distribution):

```python
from wpi_client.proto import wpi_pb2

# Target 1 gets first 8GB slice, Target 2 gets second 8GB slice
assignments = [
    wpi_pb2.ShardAssignment(
        target_node_id="10.130.0.85",
        shard_index=0,
        offset_bytes=0,
        length_bytes=8 * 1024 * 1024 * 1024,
        target_gpu_id=0
    ),
    wpi_pb2.ShardAssignment(
        target_node_id="10.130.0.86",
        shard_index=1,
        offset_bytes=8 * 1024 * 1024 * 1024,
        length_bytes=8 * 1024 * 1024 * 1024,
        target_gpu_id=0
    )
]

client.propagate(
    buffer_id=buffer_id,
    target_node_ids=["10.130.0.85", "10.130.0.86"],
    mode=1,  # 1 = SCATTER
    shard_assignments=assignments
)
```

---

### 3.5 Synchronization (Serving / Subscriber Side)
On the serving node, the application must wait until the network weight transfer completes before reading the mapped tensor.

```python
# 1. Connect to the notify socket
client.connect_notify_socket(buffer_id=buffer_id)

# 2. Block until the driver finishes receiving weights and triggers READY
logger.info("Waiting for incoming weights...")
client.wait_for_ready(timeout=120.0)

# Weights are now fully updated in VRAM; safe to execute model forward pass
logger.info("Weights updated successfully! Ready for inference.")
```

---

### 3.6 Cleaning Up (Tearing Down Allocation)
When the serving session or training job is finished, release the buffer claim to allow the WPI driver to free the VRAM.

```python
# Unstage weight and release driver allocations
client.unstage_weight(claim_id=claim_id)
client.close()
```
