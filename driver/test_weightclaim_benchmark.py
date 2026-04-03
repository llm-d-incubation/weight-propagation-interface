import json
import subprocess
import grpc
import sys
import wpi_pb2
import wpi_pb2_grpc

def parse_kubernetes_resources():
    print("Fetching WeightClaim 'wc-test' from wpi-system...")
    wc_json = subprocess.check_output(['kubectl', 'get', 'weightclaim', 'wc-test', '-n', 'wpi-system', '-o', 'json'])
    wc_data = json.loads(wc_json)
    wb_name = wc_data['spec']['weightBufferName']

    print(f"Fetching associated WeightBuffer '{wb_name}' from wpi-system...")
    wb_json = subprocess.check_output(['kubectl', 'get', 'weightbuffer', wb_name, '-n', 'wpi-system', '-o', 'json'])
    wb_data = json.loads(wb_json)
    capacity_str = wb_data['spec'].get('capacity', '')
    source_path = wb_data['spec'].get('sourcePath', '')

    print("Fetching wpi-driver pod IPs in exact order...")
    # Get exact order to match benchmark.sh's PODS array
    pods_json = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
    pods_data = json.loads(pods_json)
    
    # We must match the order of `kubectl get pods -o jsonpath='{.items[*].metadata.name}'`
    # which is the order they appear in the items list
    driver_ips = [pod.get('status', {}).get('podIP') for pod in pods_data['items'] if pod.get('status', {}).get('phase') == 'Running']

    if len(driver_ips) < 2:
        raise Exception("Need at least 2 running wpi-driver pods to benchmark internode propagation.")

    return wc_data['metadata']['name'], wb_name, capacity_str, source_path, driver_ips[0], driver_ips[1]

def test_internode_propagate():
    claim_id, buffer_id, capacity_str, source_path, ip_source, ip_target = parse_kubernetes_resources()

    # Convert capacity string to bytes
    if capacity_str.endswith('Gi'):
        size_gb = int(capacity_str.replace('Gi', ''))
        size_bytes = size_gb * 1024 * 1024 * 1024
    else:
        print(f"size_bytes is not specified: {claim_id}")
    print(f"\n--- benchmark configuration ---")
    print(f"WeightClaim: {claim_id}")
    print(f"WeightBuffer: {buffer_id}")
    print(f"SourcePath: {source_path}")
    print(f"SizeBytes: {size_bytes}")
    print(f"Source Driver Node: {ip_source}:50051")
    print(f"Target Driver Node: {ip_target}:50051")
    print("-------------------------------\n")

    print(f"Connecting to source via port-forward (localhost:50051)")
    chan_source = grpc.insecure_channel("localhost:50051")
    stub_source = wpi_pb2_grpc.NodeServiceStub(chan_source)
    
    print(f"Connecting to target via port-forward (localhost:50061)")
    chan_target = grpc.insecure_channel("localhost:50061")
    stub_target = wpi_pb2_grpc.NodeServiceStub(chan_target)

    try:
        print("\n1. Triggering DRA Staging via Pod Creation...")
        pod_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: dra-trigger-pod
  namespace: wpi-system
spec:
  nodeSelector:
    kubernetes.io/hostname: "{}"
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  resourceClaims:
  - name: wpi-device
    resourceClaimName: wc-test
  containers:
  - name: pause
    image: k8s.gcr.io/pause:3.2
    resources:
      claims:
      - name: wpi-device
"""
        # Create the pod on the SOURCE node to trigger staging
        # To find the node name for the first ip:
        nodes_json = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
        pods_data = json.loads(nodes_json)
        source_node = None
        target_node = None
        for pod in pods_data['items']:
            if pod['status']['podIP'] == ip_source:
                source_node = pod['spec']['nodeName']
            if pod['status']['podIP'] == ip_target:
                target_node = pod['spec']['nodeName']
                
        if not source_node or not target_node:
            raise Exception("Failed to find node names for the driver IPs")

        source_pod_manifest = pod_yaml.format(source_node)
        target_pod_manifest = pod_yaml.replace("dra-trigger-pod", "dra-trigger-pod-target").format(target_node)
        
        with open('/tmp/source_pod.yaml', 'w') as f:
            f.write(source_pod_manifest)
        with open('/tmp/target_pod.yaml', 'w') as f:
            f.write(target_pod_manifest)

        # Ensure any previous pod is deleted so DRA is forced to stage again
        subprocess.call(['kubectl', 'delete', 'pod', 'dra-trigger-pod', 'dra-trigger-pod-target', '-n', 'wpi-system', '--ignore-not-found'])

        subprocess.check_call(['kubectl', 'apply', '-f', '/tmp/source_pod.yaml'])
        subprocess.check_call(['kubectl', 'apply', '-f', '/tmp/target_pod.yaml'])

        print("Waiting for DRA plugin to complete NodePrepareResources on both nodes...")
        import time
        source_running = False
        target_running = False
        for _ in range(120):
            if not source_running:
                out_src = subprocess.check_output(['kubectl', 'get', 'pod', 'dra-trigger-pod', '-n', 'wpi-system', '-o', 'json'])
                if json.loads(out_src)['status'].get('phase') == 'Running':
                    source_running = True
                    print("Source pod is running! (Source Node is staged via DRA plugin)")
            
            if not target_running:
                out_tgt = subprocess.check_output(['kubectl', 'get', 'pod', 'dra-trigger-pod-target', '-n', 'wpi-system', '-o', 'json'])
                if json.loads(out_tgt)['status'].get('phase') == 'Running':
                    target_running = True
                    print("Target pod is running! (Target Node is staged via DRA plugin)")
            
            if source_running and target_running:
                break
            time.sleep(1)
        
        if not (source_running and target_running):
            raise Exception("Failed to stage both pods via DRA plugin!")

        print("\n2a. Launching consumer on target node to wait for notification...")
        out_pods = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
        wpi_pods = json.loads(out_pods)
        target_pod_name = next(p['metadata']['name'] for p in wpi_pods['items'] if p['status']['podIP'] == ip_target)
        
        subprocess.check_call(['kubectl', 'cp', './test_e2e_consumer.py', f"wpi-system/{target_pod_name}:/tmp/test_e2e_consumer.py"])
        
        consumer_proc = subprocess.Popen(
            ['kubectl', 'exec', target_pod_name, '-n', 'wpi-system', '--', 'python', '/tmp/test_e2e_consumer.py', buffer_id, str(size_gb)],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        time.sleep(3) # Wait for it to map memory and block

        print(f"\n2b. Triggering NodePropagate (Source -> Target {ip_target}:50051)...")
        request_propagate = wpi_pb2.NodePropagateRequest(
            buffer_id=buffer_id,
            target_node_ids=[ip_target]
        )
        import time
        start_time = time.time()
        stub_source.NodePropagate(request_propagate)
        end_time = time.time()
        
        latency = end_time - start_time
        transfer_size_gb = size_bytes / (1024**3)
        throughput = transfer_size_gb / latency
        print("NodePropagate command returned successfully.")
        print(f"Transfer Size: {transfer_size_gb:.2f} GB")
        print(f"Transfer Latency: {latency:.4f} seconds")
        print(f"Throughput: {throughput:.2f} GB/s")

        print(f"\n3. Waiting 3 seconds for Target async NCCL thread to finish receiving...")
        time.sleep(3)

    finally:
        print(f"\n4. Triggering DRA Unprepare on both SOURCE and TARGET by deleting pods...")
        try:
            subprocess.call(['kubectl', 'delete', 'pod', 'dra-trigger-pod', 'dra-trigger-pod-target', '-n', 'wpi-system'])
        except Exception as e:
            print(f"DRA pod delete failed: {e}")
            
        print("Cleanup complete. Memory freed.")

if __name__ == "__main__":
    test_internode_propagate()
