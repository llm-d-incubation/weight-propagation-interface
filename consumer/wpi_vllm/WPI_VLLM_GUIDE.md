# WPI + vLLM Integration Guide

This guide describes how to set up, configure, and run benchmarks for the Weight Propagation Interface (WPI) integration with vLLM on a Google Kubernetes Engine (GKE) cluster.

## Overview

The Weight Propagation Interface (WPI) enables high-performance, zero-copy model weight distribution for ML inference workloads on Kubernetes. This integration allows vLLM to receive weights directly from a trainer or a source pod via zero-copy VRAM mapping, bypassing standard network bottlenecks and CPU-to-GPU transfers.

## Prerequisites

Before you begin, ensure the following are set up in your cluster:

1.  **WPI Driver Deployed**: The `wpi-driver` DaemonSet must be running in the `wpi-system` namespace.
2.  **CRDs Installed**: Custom Resource Definitions for `WeightBuffer` and `WeightClaim` must be installed with the API group `wpi.sig.k8s.io/v1alpha1`.
3.  **Hugging Face Token**: A Kubernetes secret named `hf-token-secret` must be created in the `wpi-system` namespace containing your token:
    ```bash
    kubectl create secret generic hf-token-secret --from-literal=token='<YOUR_HF_TOKEN>' -n wpi-system
    ```
4.  **Model Access**: If using gated models like Llama 3, ensure the account associated with your Hugging Face token has been granted access by Meta.
5.  **WPI Client Library**: If you need to install the client library in a custom environment, you can install it directly from GitHub:
    ```bash
    pip install git+https://github.com/llm-d-incubation/weight-propagation-interface.git#subdirectory=consumer/wpi_client
    ```

## Configuration Files

The integration uses the following manifests located in `consumer/wpi_vllm/deploy/`:

*   **`wpi_resources.yaml`**: Defines the `WeightBuffer` and `WeightClaim` resources.
*   **`vllm_server.yaml`**: Deploys the vLLM inference pod. Ensure it uses an available node pool (e.g., `a4-pool`) and that the image path is correct.
*   **`trainer.yaml`**: Deploys a pod to simulate the trainer and run benchmarks.

## Execution (Automated Cluster Run)

To run the benchmark inside the cluster and avoid host resource limits, use the provided automation script:

1.  Navigate to the benchmark directory:
    ```bash
    cd consumer/wpi_vllm/benchmark
    ```
2.  Run the script specifying a non-gated model (like Qwen) or an authorized model:
    ```bash
    ./run_cluster_benchmark.sh Qwen/Qwen2-7B
    ```
3.  **Monitor Progress**: Check the logs of the trainer pod to see results:
    ```bash
    kubectl logs wpi-vllm-trainer -n wpi-system
    ```

## Performance Results

During initial validation on A4 nodes, the following performance was observed for the `Qwen/Qwen2-7B` model (14.185 GB weights):

*   **Mean Bandwidth**: **~20.73 GB/s** (peaking at **~20.99 GB/s**).
*   **Mean Total Time**: **~684.38 ms** for end-to-end weight update.

## Troubleshooting

*   **`401 Unauthorized`**: Missing or invalid Hugging Face token in the secret.
*   **`404 Not Found`**: Incorrect model identifier in manifest. (e.g., `meta-llama/Llama-3-8B` instead of `Meta-Llama-3-8B`).
*   **`403 Forbidden`**: The account has not been granted access to the requested gated model. Try using a non-gated model like `Qwen/Qwen2-7B` as a fallback.
*   **`ModuleNotFoundError: No module named 'vllm'`**: Occurs when running benchmark scripts on the host without installed dependencies. Always prefer running in-cluster using the `trainer.yaml` pod.
*   **`python: command not found`**: The container images only have `python3` available in the PATH. Manifests must use `python3`.
