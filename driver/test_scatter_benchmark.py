import json
import subprocess
import grpc
import sys
import wpi_pb2
import wpi_pb2_grpc
import time

def get_driver_pods():
    print("Fetching wpi-driver pod IPs...")
    pods_json = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
    pods_data = json.loads(pods_json)
    running_pods = [pod for pod in pods_data['items'] if pod.get('status', {}).get('phase') == 'Running']
    
    if len(running_pods) < 2:
        raise Exception("Need at least 2 running wpi-driver pods to benchmark scatter propagation.")

    # We need to know IPs and Node names
    source_ip = running_pods[0]['status']['podIP']
    source_node = running_pods[0]['spec']['nodeName']
    
    target_ips_and_nodes = [(pod['status']['podIP'], pod['spec']['nodeName']) for pod in running_pods[1:]]
    return source_ip, source_node, target_ips_and_nodes

def test_scatter_propagate():
    buffer_id = "wb-test-sharded"
    source_ip, source_node, target_ips_and_nodes = get_driver_pods()
    
    # We will test scattering to 1 target: node 1 gets shard 0.
    # If there are >1 targets, let's say node 1 gets shard 0, node 2 gets shard 1.
    target_ip, target_node = target_ips_and_nodes[0]
    
    size_bytes = 600 * 1024 * 1024 * 1024 # 600 GB
    shard_size = size_bytes // 8 # 75 GB
    
    print(f"\n--- benchmark configuration ---")
    print(f"WeightBuffer: {buffer_id}")
    print(f"Total Size: {size_bytes} bytes")
    print(f"Shard Size: {shard_size} bytes")
    print(f"Source Node: {source_node} ({source_ip}:50051)")
    print(f"Target Node: {target_node} ({target_ip}:50051)")
    print("-------------------------------\n")

    print(f"Connecting to source via port-forward (localhost:50051)")
    chan_source = grpc.insecure_channel("localhost:50051")
    stub_source = wpi_pb2_grpc.NodeServiceStub(chan_source)

    try:
        print("\n1. Triggering DRA Staging via Pod Creation...")
        pod_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: wpi-system
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
        # Source pod claims nothing or claims shard 0 (we just want the buffer staged)
        source_pod_manifest = pod_yaml.format(pod_name="dra-trigger-pod-src", node_name=source_node, claim_name="wc-test-sharded-0")
        target_pod_manifest = pod_yaml.format(pod_name="dra-trigger-pod-tgt", node_name=target_node, claim_name="wc-test-sharded-1")
        
        with open('/tmp/source_pod_scatter.yaml', 'w') as f:
            f.write(source_pod_manifest)
        with open('/tmp/target_pod_scatter.yaml', 'w') as f:
            f.write(target_pod_manifest)

        subprocess.call(['kubectl', 'delete', 'pod', 'dra-trigger-pod-src', 'dra-trigger-pod-tgt', '-n', 'wpi-system', '--ignore-not-found'])

        subprocess.check_call(['kubectl', 'apply', '-f', '/tmp/source_pod_scatter.yaml'])
        subprocess.check_call(['kubectl', 'apply', '-f', '/tmp/target_pod_scatter.yaml'])

        print("Waiting for DRA plugin to complete NodePrepareResources on both nodes...")
        source_running = False
        target_running = False
        for _ in range(120):
            if not source_running:
                out_src = subprocess.check_output(['kubectl', 'get', 'pod', 'dra-trigger-pod-src', '-n', 'wpi-system', '-o', 'json'])
                if json.loads(out_src)['status'].get('phase') == 'Running':
                    source_running = True
            
            if not target_running:
                out_tgt = subprocess.check_output(['kubectl', 'get', 'pod', 'dra-trigger-pod-tgt', '-n', 'wpi-system', '-o', 'json'])
                if json.loads(out_tgt)['status'].get('phase') == 'Running':
                    target_running = True
            
            if source_running and target_running:
                print("Both pods are running! DRA staging completed.")
                break
            time.sleep(1)
        
        if not (source_running and target_running):
            raise Exception("Failed to stage pods.")

        print("\n2a. Launching consumer on target node to wait for notification...")
        out_pods = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
        wpi_pods = json.loads(out_pods)
        target_driver_pod = next(p['metadata']['name'] for p in wpi_pods['items'] if p['status']['podIP'] == target_ip)
        
        # We need the effective buffer ID because the target claimed shard 1
        effective_buffer_id = f"{buffer_id}__shard_1"
        
        subprocess.check_call(['kubectl', 'cp', './test_e2e_consumer.py', f"wpi-system/{target_driver_pod}:/tmp/test_e2e_consumer.py"])
        
        # Test consumer mapped the 1 GB shard
        consumer_proc = subprocess.Popen(
            ['kubectl', 'exec', target_driver_pod, '-n', 'wpi-system', '--', 'python3', '/tmp/test_e2e_consumer.py', effective_buffer_id, "1"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        time.sleep(3)

        print(f"\n2b. Triggering NodePropagate (SCATTER mode)...")
        # In scatter mode, we assign shard 1 (offset 0 because Source only holds shard 0 locally)
        assignment = wpi_pb2.ShardAssignment(
            target_node_id=target_ip,
            shard_index=1,
            offset_bytes=0, # Local offset in Source buffer
            length_bytes=shard_size,
            target_gpu_id=0
        )
        
        request_propagate = wpi_pb2.NodePropagateRequest(
            buffer_id=f"{buffer_id}__shard_0",
            target_node_ids=[target_ip],
            mode=1, # SCATTER mode
            shard_assignments=[assignment]
        )
        start_time = time.time()
        stub_source.NodePropagate(request_propagate)
        end_time = time.time()
        
        latency = end_time - start_time
        transfer_size_gb = shard_size / (1024**3)
        throughput = transfer_size_gb / latency
        print("NodePropagate command returned successfully.")
        print(f"Scattered Shard Size: {transfer_size_gb:.2f} GB")
        print(f"Scatter Latency: {latency:.4f} seconds")
        print(f"Throughput: {throughput:.2f} GB/s")

        print(f"\n3. Waiting 3 seconds for Target async NCCL thread to finish receiving...")
        time.sleep(3)

    finally:
        print(f"\n4. Cleaning up pods...")
        try:
            subprocess.call(['kubectl', 'delete', 'pod', 'dra-trigger-pod-src', 'dra-trigger-pod-tgt', '-n', 'wpi-system'])
        except Exception:
            pass
        print("Cleanup complete.")

if __name__ == "__main__":
    test_scatter_propagate()
