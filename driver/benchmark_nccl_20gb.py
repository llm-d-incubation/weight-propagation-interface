import cupy
import cupy.cuda.nccl as nccl
import time
import multiprocessing as mp

def run_target(nccl_id):
    print("Child Target started")
    cupy.cuda.Device(0).use()
    comm = nccl.NcclCommunicator(2, nccl_id, 1)
    
    # Allocate 18GB buffer
    size_bytes = 18 * 1024 * 1024 * 1024
    num_elements = size_bytes // 2
    
    arr = cupy.zeros(num_elements, dtype=cupy.float16)
    
    # Warmup
    comm.recv(arr.data.ptr, 1, nccl.NCCL_FLOAT16, 0, 0)
    cupy.cuda.Device(0).synchronize()
    
    # Target recv
    print("Target ready to receive 20GB.")
    start_time = time.time()
    comm.recv(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 0, 0)
    cupy.cuda.Device(0).synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    bandwidth = (size_bytes / (1024**3)) / duration
    print(f"Target Recv Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")

def run_source(nccl_id):
    print("Child Source started")
    cupy.cuda.Device(0).use()
    comm = nccl.NcclCommunicator(2, nccl_id, 0)
    
    size_bytes = 18 * 1024 * 1024 * 1024
    num_elements = size_bytes // 2
    
    arr = cupy.zeros(num_elements, dtype=cupy.float16)
    
    # Warmup
    comm.send(arr.data.ptr, 1, nccl.NCCL_FLOAT16, 1, 0)
    cupy.cuda.Device(0).synchronize()
    
    # Source send
    print("Source ready to send 20GB.")
    start_time = time.time()
    comm.send(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 1, 0)
    cupy.cuda.Device(0).synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    bandwidth = (size_bytes / (1024**3)) / duration
    print(f"Source Send Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    nccl_id = nccl.get_unique_id()
    
    p1 = mp.Process(target=run_target, args=(nccl_id,))
    p2 = mp.Process(target=run_source, args=(nccl_id,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
