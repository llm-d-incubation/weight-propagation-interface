import grpc
import os
import wpi_pb2
import wpi_pb2_grpc

def test_stage_weight():
    channel = grpc.insecure_channel('localhost:50051')
    stub = wpi_pb2_grpc.NodeServiceStub(channel)
    
    print("Calling NodeStageWeight...")
    request = wpi_pb2.NodeStageWeightRequest(
        claim_id="test-claim-id",
        buffer_id="test-buffer-id"
    )
    
    try:
        response = stub.NodeStageWeight(request)
        print("NodeStageWeight call successful!")
        
        file_path = "/dev/wpi/weights/layer82"
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"File {file_path} exists. Size: {size / (1024**3):.2f} GiB")
            if size == 2 * 1024 * 1024 * 1024:
                print("Size matches 2 GiB exactly!")
            else:
                print(f"Size mismatch! Expected 2GiB, got {size} bytes")
        else:
            print(f"File {file_path} NOT found!")
            
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    test_stage_weight()
