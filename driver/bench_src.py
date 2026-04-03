import cupy
import cupy.cuda.nccl as nccl
import time
import os

cupy.cuda.Device(0).use()
nccl_id = nccl.get_unique_id()
with open('/app/nccl_id.txt', 'w') as f:
    f.write(','.join(str(x) for x in nccl_id))

print(f"Source Rank 0 generated ID.")

comm = nccl.NcclCommunicator(2, nccl_id, 0)
print("Source comm init done")

size_bytes = 20 * 1024 * 1024 * 1024
num_elements = size_bytes // 2
arr = cupy.zeros(num_elements, dtype=cupy.float16)
print("Source array allocated")

comm.send(arr.data.ptr, 1, nccl.NCCL_FLOAT16, 1, 0)
cupy.cuda.Device(0).synchronize()

print("Source ready to send 20GB.")
start_time = time.time()
comm.send(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 1, 0)
cupy.cuda.Device(0).synchronize()
end_time = time.time()

duration = end_time - start_time
bandwidth = (size_bytes / (1024**3)) / duration
print(f"Source Send Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")
