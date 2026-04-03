import sys
import cupy
import cupy.cuda.nccl as nccl
import time

def run_target(nccl_id_val):
    print("Target started")
    cupy.cuda.Device(1).use()
    nccl_id = bytes(int(x) for x in nccl_id_val.split(','))
    print(f"nccl_id length: {len(nccl_id)}")
    print("Initializing communicator...")
    comm = nccl.NcclCommunicator(2, nccl_id, 1)
    print("Communicator initialized.")
    
    size_bytes = 10 * 1024 * 1024 * 1024
    num_elements = size_bytes // 2
    
    arr = cupy.zeros(num_elements, dtype=cupy.float16)
    
    print("Target ready to receive.")
    start_time = time.time()
    comm.recv(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 0, 0)
    cupy.cuda.Device(1).synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    bandwidth = (size_bytes / (1024**3)) / duration
    print(f"Target Recv Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    run_target(sys.argv[1])
