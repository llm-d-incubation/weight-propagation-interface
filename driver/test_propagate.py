import grpc
import os
import wpi_pb2
import wpi_pb2_grpc

def test_propagate():
    channel = grpc.insecure_channel('localhost:50051')
    stub = wpi_pb2_grpc.NodeServiceStub(channel)
    
    # 1. Stage the weight so that it allocates VRAM and records the buffer_id
    print("Calling NodeStageWeight...")
    request_stage = wpi_pb2.NodeStageWeightRequest(
        claim_id="test-claim-id",
        buffer_id="test-buffer-id"
    )
    
    try:
        stub.NodeStageWeight(request_stage)
        print("NodeStageWeight call successful!")
    except grpc.RpcError as e:
        print(f"NodeStageWeight Error: {e.code()} - {e.details()}")
        return

    # 2. Trigger NodePropagate back to localhost (same daemon)
    # This will cause the single local node to exchange the handshake acting as both Sender and Receiver.
    print("Calling NodePropagate...")
    request_propagate = wpi_pb2.NodePropagateRequest(
        buffer_id="test-buffer-id",
        target_node_id="127.0.0.1"
    )
    
    try:
        stub.NodePropagate(request_propagate)
        print("NodePropagate call successful! The NCCL transfer completed via local loopback.")
    except grpc.RpcError as e:
        print(f"NodePropagate Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    test_propagate()
