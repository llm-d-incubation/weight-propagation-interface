import grpc
import sys
import subprocess
import json
import wpi_pb2
import wpi_pb2_grpc

def test_internode_propagate():
    print("Fetching wpi-driver pod IPs...")
    pods_json = subprocess.check_output(['kubectl', 'get', 'pods', '-l', 'app=wpi-driver', '-n', 'wpi-system', '-o', 'json'])
    pods_data = json.loads(pods_json)
    driver_ips = [pod['status']['podIP'] for pod in pods_data['items'] if pod['status'].get('phase') == 'Running']
    if len(driver_ips) < 2:
        raise Exception("Need at least 2 running wpi-driver pods.")
        
    ip_source = driver_ips[0]
    ip_target = driver_ips[1]

    print(f"Connecting to source: localhost:50051 (Forwarded from {ip_source}:50051)")
    chan_source = grpc.insecure_channel('localhost:50051')
    stub_source = wpi_pb2_grpc.NodeServiceStub(chan_source)
    
    print(f"Connecting to target: localhost:50061 (Forwarded from {ip_target}:50051)")
    chan_target = grpc.insecure_channel('localhost:50061')
    stub_target = wpi_pb2_grpc.NodeServiceStub(chan_target)
    
    buffer_id = "wb-test"
    
    print("\n1. Calling NodeStageWeight on SOURCE (50051)...")
    request_stage = wpi_pb2.NodeStageWeightRequest(
        claim_id="wc-test",
        buffer_id=buffer_id,
        source_path="/etc/hostname",
        size_bytes=10 * 1024 * 1024 * 1024
    )
    stub_source.NodeStageWeight(request_stage)
    print("Source staging complete.")
    
    print("\n2. Calling NodeStageWeight on TARGET (50061)...")
    stub_target.NodeStageWeight(request_stage)
    print("Target staging complete.")

    print(f"\n3. Triggering NodePropagate (Source -> Target {ip_target}:50051)...")
    request_propagate = wpi_pb2.NodePropagateRequest(
        buffer_id=buffer_id,
        target_node_ids=[ip_target]
    )
    
    import time
    
    num_warmup = 5
    num_iterations = 10
    
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        stub_source.NodePropagate(request_propagate)
        
    print(f"Running {num_iterations} active iterations...")
    start = time.time()
    for _ in range(num_iterations):
        stub_source.NodePropagate(request_propagate)
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    # 10 GiB payload
    bandwidth = 10.0 / avg_time 
    
    print("NodePropagate command returned successfully.")
    print(f"Total time for {num_iterations} iterations: {end-start:.4f}s")
    print(f"Average Transfer time: {avg_time:.4f}s")
    print(f"Average Bandwidth: {bandwidth:.2f} GB/s")

    print(f"\n4. Calling NodeUnstageWeight on SOURCE and TARGET...")
    request_unstage = wpi_pb2.NodeUnstageWeightRequest(buffer_id=buffer_id)
    stub_source.NodeUnstageWeight(request_unstage)
    stub_target.NodeUnstageWeight(request_unstage)
    print("Cleanup complete. Memory freed.")

if __name__ == "__main__":
    test_internode_propagate()
