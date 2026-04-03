import cupy
import cupy.cuda.nccl as nccl
import time
import os

cupy.cuda.Device(1).use()

while not os.path.exists('/app/nccl_id.txt'):
    time.sleep(0.1)

time.sleep(2) # Give Rank 0 time to reach init
with open('/app/nccl_id.txt', 'r') as f:
    nccl_id_val = f.read().strip()

nccl_id = bytes(int(x) for x in nccl_id_val.split(','))
print("Target Rank 1 read ID")

comm = nccl.NcclCommunicator(2, nccl_id, 1)
print("Target comm init done")

size_bytes = 20 * 1024 * 1024 * 1024
num_elements = size_bytes // 2
arr = cupy.zeros(num_elements, dtype=cupy.float16)
print("Target array allocated")

comm.recv(arr.data.ptr, 1, nccl.NCCL_FLOAT16, 0, 0)
cupy.cuda.Device(1).synchronize()

print("Target ready to receive 20GB.")
start_time = time.time()
comm.recv(arr.data.ptr, num_elements, nccl.NCCL_FLOAT16, 0, 0)
cupy.cuda.Device(1).synchronize()
end_time = time.time()

duration = end_time - start_time
bandwidth = (size_bytes / (1024**3)) / duration
print(f"Target Recv Duration: {duration:.4f}s. Bandwidth: {bandwidth:.2f} GB/s")
