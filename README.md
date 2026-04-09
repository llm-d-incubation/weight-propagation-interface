# Weight Propagation Interface (WPI)

The **Weight Propagation Interface (WPI)** is a Kubernetes-native orchestration framework designed to enable high-speed, zero-copy movement of large ML model weights between AI accelerators (GPUs, TPUs) across nodes in a cluster.

As models grow to hundreds of billions of parameters, the traditional path of saving weights to shared storage and independently downloading them into GPU RAM becomes a severe bottleneck. WPI solves this by treating Model Weights as first-class scheduling and hardware resources, leveraging native hardware interconnects (like NVLink and InfiniBand via NCCL) to securely and efficiently distribute weights directly into accelerator memory.

## 📺 Demo

Check out the [WPI Demo Video](https://youtu.be/I3bRaShPILE) to see the system in action!

## 🏗️ Architecture

WPI consists of three main architectural layers:

1.  **Custom Resource Definitions (CRDs):** Define logical blocks of weights (`WeightBuffer`) and how workloads bind to them (`WeightClaim`).
2.  **WPI Operator (The Brain):** A Kubernetes controller that reconciles the desired distribution of weights with the cluster's physical topology.
3.  **WPI Driver / Node Agent (The Mover):** A privileged daemonset running on accelerator nodes that executes hardware-specific commands (CUDA IPC, NCCL) to allocate, share, and transmit memory.
4.  **Consumer (The Workload):** The ML framework (e.g., PyTorch, vLLM) that natively binds to the shared weight memory without allocating a duplicate copy.

## 📂 Repository Structure

- `operator/`: Kubernetes controller for WPI.
- `driver/`: Node agent (Python controller and Go-based DRA plugin).
- `proto/`: gRPC service definitions (`wpi.proto`).
- `crds/`: Kubernetes Custom Resource manifests.
- `consumer/`: Example workloads and pod specifications demonstrating WPI integration.

## 🚀 Getting Started

Check out the following documents for more details:

- [WPI Design Document](WPI_DESIGN.md): Detailed architectural design.
- [WPI User Guide](WPI_USER_GUIDE.md): Setup and usage instructions.

Use `setup.sh` to initialize the environment and `teardown.sh` to clean up.
