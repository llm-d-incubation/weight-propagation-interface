import sys
import cupy
import cupy.cuda.nccl as nccl
import time

def run_source(nccl_id_val):
    print("Source started")
    cupy.cuda.Device(0).use()
    nccl_id = bytes(int(x) for x in nccl_id_val.split(','))
    print(f"nccl_id length: {len(nccl_id)}")
    print("Initializing communicator...")
    comm = nccl.NcclCommunicator(2, nccl_id, 0)
    print("Communicator initialized.")
    
    size_bytes = 10 * 1024 * 1024 * 1024
    num_elements = size_bytes // 2
    
    arr = cupy.zeros(num_elements, dtype=cupy.float16)
    
    print("Source ready to send.")
    start_time = time.time()
    comm.send(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 1, 0)
    cupy.cuda.Device(0).synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    bandwidth = (size_bytes / (1024**3)) / duration
    print(f"Source Send Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_source(sys.argv[1])
    else:
        nccl_id = nccl.get_unique_id()
        nccl_id_str = ",".join(str(x) for x in nccl_id)
        print("NCCL_ID=" + nccl_id_str)
