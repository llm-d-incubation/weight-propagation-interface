import grpc
import sys
import wpi_pb2
import wpi_pb2_grpc

def stage():
    channel = grpc.insecure_channel("localhost:50051")
    stub = wpi_pb2_grpc.NodeServiceStub(channel)
    req = wpi_pb2.NodeStageWeightRequest(
        claim_id="wc-test",
        buffer_id="wb-test",
        size_bytes=10*1024*1024*1024,
        source_path="/etc/hostname"
    )
    stub.NodeStageWeight(req)
    print("Successfully staged wb-test on localhost")

if __name__ == "__main__":
    stage()
