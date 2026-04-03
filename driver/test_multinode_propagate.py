import grpc
import sys
import wpi_pb2
import wpi_pb2_grpc

def test_internode_propagate():
    print("Connecting to source: 192.168.0.7:50051")
    chan_source = grpc.insecure_channel('192.168.0.7:50051')
    stub_source = wpi_pb2_grpc.NodeServiceStub(chan_source)
    
    print("Connecting to target 1: 192.168.0.6:50051")
    chan_target_1 = grpc.insecure_channel('192.168.0.6:50051')
    stub_target_1 = wpi_pb2_grpc.NodeServiceStub(chan_target_1)

    print("Connecting to target 2: 192.168.0.5:50051")
    chan_target_2 = grpc.insecure_channel('192.168.0.5:50051')
    stub_target_2 = wpi_pb2_grpc.NodeServiceStub(chan_target_2)
    
    buffer_id = "test-20gb-buffer"
    
    print("\n1. Calling NodeStageWeight on SOURCE (50051)...")
    request_stage = wpi_pb2.NodeStageWeightRequest(
        claim_id="claim-20gb",
        buffer_id=buffer_id
    )
    stub_source.NodeStageWeight(request_stage)
    print("Source staging complete.")
    
    print("\n2. Calling NodeStageWeight on TARGET 1...")
    stub_target_1.NodeStageWeight(request_stage)
    print("Target 1 staging complete.")

    print("\n2b. Calling NodeStageWeight on TARGET 2...")
    stub_target_2.NodeStageWeight(request_stage)
    print("Target 2 staging complete.")

    print("\n3. Triggering NodePropagate (Source -> Target1, Target2)...")
    request_propagate = wpi_pb2.NodePropagateRequest(
        buffer_id=buffer_id,
        target_node_ids=["192.168.0.6", "192.168.0.5"]
    )
    stub_source.NodePropagate(request_propagate)
    print("NodePropagate command returned successfully.")

if __name__ == "__main__":
    test_internode_propagate()
