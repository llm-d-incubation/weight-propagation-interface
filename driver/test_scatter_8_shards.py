import json
import subprocess
import grpc
import sys
import wpi_pb2
import wpi_pb2_grpc
import time
import concurrent.futures

def get_driver_pods():
    print("Fetching wpi-driver pod IPs...")
    pods_json = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
    pods_data = json.loads(pods_json)
    running_pods = [pod for pod in pods_data['items'] if pod.get('status', {}).get('phase') == 'Running']
    
    if len(running_pods) < 2:
        raise Exception("Need at least 2 running wpi-driver pods to benchmark scatter propagation.")

    source_ip = running_pods[0]['status']['podIP']
    source_node = running_pods[0]['spec']['nodeName']
    source_pod = running_pods[0]['metadata']['name']
    
    target_ips_and_nodes = [(pod['status']['podIP'], pod['spec']['nodeName']) for pod in running_pods[1:]]
    return source_ip, source_node, source_pod, target_ips_and_nodes

def apply_pod(pod_name, node_name, claim_name):
    pod_yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: wpi-system
  labels:
    benchmark: "8shard"
spec:
  nodeSelector:
    kubernetes.io/hostname: "{node_name}"
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  resourceClaims:
  - name: wpi-device
    resourceClaimName: {claim_name}
  containers:
  - name: pause
    image: k8s.gcr.io/pause:3.2
    resources:
      claims:
      - name: wpi-device
"""
    file_path = f'/tmp/{pod_name}.yaml'
    with open(file_path, 'w') as f:
        f.write(pod_yaml)
    subprocess.check_call(['kubectl', 'apply', '-f', file_path])

def execute_propagate(stub, buffer_id, target_ip, shard_index, shard_size):
    """Propagate a single shard from source GPU to matching target GPU."""
    assignment = wpi_pb2.ShardAssignment(
        target_node_id=target_ip,
        shard_index=shard_index,
        offset_bytes=0,
        length_bytes=shard_size,
        target_gpu_id=0
    )
    request = wpi_pb2.NodePropagateRequest(
        buffer_id=f"{buffer_id}__shard_{shard_index}",
        target_node_ids=[target_ip],
        mode=1,
        shard_assignments=[assignment]
    )
    start_time = time.time()
    stub.NodePropagate(request)
    end_time = time.time()
    latency = end_time - start_time
    gb = shard_size / (1024**3)
    print(f"  Shard {shard_index}: {gb:.0f} GB in {latency:.4f}s ({gb/latency:.2f} GB/s)")
    return latency

def test_8_shard_propagate():
    buffer_id = "wb-test-sharded"
    source_ip, source_node, source_pod, target_ips_and_nodes = get_driver_pods()
    target_ip, target_node = target_ips_and_nodes[0]
    
    size_bytes = 600 * 1024 * 1024 * 1024  # 600 GB total
    shard_size = size_bytes // 8            # 75 GB per shard
    num_shards = 8
    
    # All 8 shards staged on BOTH nodes — full 8-GPU utilization
    shard_indices = list(range(num_shards))
    pod_names = []
    total_pods = num_shards * 2  # 8 source + 8 target

    print(f"\n--- 8-Shard 600GB Full 8-GPU Benchmark ---")
    print(f"Source Node: {source_node}")
    print(f"Target Node: {target_node}")
    print(f"Shards: {num_shards} x {shard_size / (1024**3):.0f} GB = {size_bytes / (1024**3):.0f} GB")
    
    try:
        print(f"\n1. Spawning {total_pods} DRA pods to map all shards on both nodes...")
        
        # Stage all 8 shards on source node
        for i in shard_indices:
            pod_name = f"dra-src-{i}"
            apply_pod(pod_name, source_node, f"wc-test-sharded-src-{i}")
            pod_names.append(pod_name)
        
        # Stage all 8 shards on target node
        for i in shard_indices:
            pod_name = f"dra-tgt-{i}"
            apply_pod(pod_name, target_node, f"wc-test-sharded-tgt-{i}")
            pod_names.append(pod_name)
            
        print(f"Waiting for all {total_pods} DRA plugins to bind VRAM mappings ({size_bytes // (1024**3)} GB per node)...")
        running = []
        for _ in range(180):
            out = subprocess.check_output(['kubectl', 'get', 'pods', '-n', 'wpi-system', '-o', 'json'])
            pods = json.loads(out)
            running = [p for p in pods['items'] if p['metadata']['name'] in pod_names and p['status'].get('phase') == 'Running']
            if len(running) == total_pods:
                print(f"All {total_pods} pods are Running! Full 8-GPU matrix mapped on both nodes.")
                break
            time.sleep(1)
            
        if len(running) < total_pods:
            print(f"WARNING: Only {len(running)}/{total_pods} pods running")
            raise Exception(f"Failed to stage all {total_pods} shards concurrently.")

        print(f"\n2. Executing {num_shards} Concurrent Stream Propagations ({size_bytes // (1024**3)} GB Aggregate)...")
        pf_proc = subprocess.Popen(['kubectl', 'port-forward', f'pod/{source_pod}', '50051:50051', '-n', 'wpi-system'])
        time.sleep(3)
        chan_source = grpc.insecure_channel("localhost:50051")
        stub_source = wpi_pb2_grpc.NodeServiceStub(chan_source)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
            futures = []
            for i in shard_indices:
                futures.append(executor.submit(
                    execute_propagate, stub_source, buffer_id, target_ip, i, shard_size
                ))
                
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        avg_latency = sum(results) / len(results)
        max_latency = max(results)
        total_gb = num_shards * (shard_size / (1024**3))
        aggregate_bandwidth = total_gb / max_latency  # wall-clock throughput
        avg_per_stream = (shard_size / (1024**3)) / avg_latency
        
        print(f"\n--- Benchmark Complete ---")
        print(f"Propagated {total_gb:.0f} GB total across {num_shards} concurrent NCCL streams!")
        print(f"Average Stream Latency: {avg_latency:.4f} seconds")
        print(f"Max Stream Latency (wall clock): {max_latency:.4f} seconds")
        print(f"Per-Stream Bandwidth (avg): {avg_per_stream:.2f} GB/s")
        print(f"Aggregate Cross-Node Throughput: {aggregate_bandwidth:.2f} GB/s")
    
    finally:
        if 'pf_proc' in locals():
            pf_proc.terminate()
        print("\n3. Tearing down 8-shard matrix skipped for debugging...")
        pass

if __name__ == "__main__":
    test_8_shard_propagate()
