import socket
import ctypes
import array
import os
import sys
import time
import cupy

CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int)
    ]

def get_mapped_ptr(buffer_id):
    sock_path = f"/run/wpi/sockets/{buffer_id}.sock"
    
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    
    fds = array.array("i", [0])
    msg, ancdata, flags, addr = client.recvmsg(1, socket.CMSG_LEN(fds.itemsize))
    
    fd = -1
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            fd = fds[1]
            
    if fd == -1 or fd == 0:
        print("Failed to receive File Descriptor")
        sys.exit(1)
        
    libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
    libcuda.cuInit(0)
    device = ctypes.c_int()
    libcuda.cuDeviceGet(ctypes.byref(device), 0)
    ctx = ctypes.c_void_p()
    libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
    
    handle = ctypes.c_ulonglong()
    libcuda.cuMemImportFromShareableHandle(ctypes.byref(handle), ctypes.c_void_p(fd), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    
    size_bytes = 10 * 1024 * 1024 * 1024 
    
    device_ptr = ctypes.c_ulonglong()
    libcuda.cuMemAddressReserve(ctypes.byref(device_ptr), ctypes.c_size_t(size_bytes), ctypes.c_size_t(0), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
    libcuda.cuMemMap(device_ptr, ctypes.c_size_t(size_bytes), ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0))
    
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device.value
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    
    libcuda.cuMemSetAccess(device_ptr, ctypes.c_size_t(size_bytes), ctypes.byref(desc), ctypes.c_size_t(1))
    
    return device_ptr.value, size_bytes

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_live_update.py <buffer-id> <role: reader|writer>")
        sys.exit(1)
        
    buffer_id = sys.argv[1]
    role = sys.argv[2]
    
    ptr, size = get_mapped_ptr(buffer_id)
    
    mem = cupy.cuda.UnownedMemory(ptr, size, None)
    memptr = cupy.cuda.MemoryPointer(mem, 0)
    
    # Map the first 5 elements as float16
    tensor = cupy.ndarray((5,), dtype=cupy.float16, memptr=memptr)
    
    if role == "reader":
        print("Starting continuous read...")
        for _ in range(5):
            print(f"Current Tensor Values: {tensor}")
            time.sleep(1)
            
    elif role == "writer":
        print(f"Original Tensor: {tensor}")
        print("Writing new values to shared VRAM [7.0, 7.0, 7.0, 7.0, 7.0]...")
        tensor[:] = cupy.array([7.0, 7.0, 7.0, 7.0, 7.0], dtype=cupy.float16)
        cupy.cuda.Device(0).synchronize()
        print(f"Updated Tensor: {tensor}")

if __name__ == "__main__":
    main()
