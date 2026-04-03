import socket
import ctypes
import array
import os
import sys

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

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_e2e_consumer.py <buffer-id> <size_gb>")
        sys.exit(1)
        
    buffer_id = sys.argv[1]
    size_gb = int(sys.argv[2])
    size_bytes = size_gb * 1024 * 1024 * 1024
    sock_path = f"/run/wpi/sockets/{buffer_id}.sock"
    
    print(f"Connecting to WPI socket: {sock_path}")
    if not os.path.exists(sock_path):
        print(f"Socket does not exist!")
        sys.exit(1)
        
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    
    print("Waiting for File Descriptor...")
    fds = array.array("i", [0])
    msg, ancdata, flags, addr = client.recvmsg(1, socket.CMSG_LEN(fds.itemsize))
    
    fd = -1
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            fd = fds[1] # <--- FIX: fds[0] is the initial 0, fds[1] is the appended FD
            
    if fd == -1 or fd == 0:
        print("Failed to receive File Descriptor")
        sys.exit(1)
        
    print(f"✅ Successfully received File Descriptor: {fd}")
    
    print("Initializing CUDA and mapping memory...")
    libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
    if libcuda.cuInit(0) != 0:
        print("cuInit failed")
        sys.exit(1)
        
    device = ctypes.c_int()
    libcuda.cuDeviceGet(ctypes.byref(device), 0)
    
    ctx = ctypes.c_void_p()
    libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
    
    handle = ctypes.c_ulonglong()
    res = libcuda.cuMemImportFromShareableHandle(
        ctypes.byref(handle),
        ctypes.c_void_p(fd),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    if res != 0:
        print(f"cuMemImportFromShareableHandle failed with code {res}")
        sys.exit(1)
        
    device_ptr = ctypes.c_ulonglong()
    res = libcuda.cuMemAddressReserve(
        ctypes.byref(device_ptr),
        ctypes.c_size_t(size_bytes),
        ctypes.c_size_t(0),
        ctypes.c_ulonglong(0),
        ctypes.c_ulonglong(0)
    )
    if res != 0:
        print(f"cuMemAddressReserve failed with code {res}")
        sys.exit(1)
        
    res = libcuda.cuMemMap(
        device_ptr,
        ctypes.c_size_t(size_bytes),
        ctypes.c_size_t(0),
        handle,
        ctypes.c_ulonglong(0)
    )
    if res != 0:
        print(f"cuMemMap failed with code {res}")
        sys.exit(1)
        
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device.value
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    
    res = libcuda.cuMemSetAccess(
        device_ptr,
        ctypes.c_size_t(size_bytes),
        ctypes.byref(desc),
        ctypes.c_size_t(1)
    )
    if res != 0:
        print(f"cuMemSetAccess failed with code {res}")
        sys.exit(1)
        
    print(f"✅ Successfully mapped {size_bytes} bytes of VRAM at device pointer {hex(device_ptr.value)}")
    print("✨ End-to-End Test Passed: Application successfully mapped the WPI WeightClaim!")
    
    notify_sock_path = f"/run/wpi/sockets/{buffer_id}_notify.sock"
    if os.path.exists(notify_sock_path):
        print(f"Connecting to Notification socket: {notify_sock_path}")
        client_notify = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_notify.connect(notify_sock_path)
        print("Listening for VRAM weight updates from driver...")
        while True:
            data = client_notify.recv(1024)
            if not data:
                print("Notification socket closed by driver.")
                break
            if b"READY" in data:
                print("✅ Received READY signal: Weights have been fully updated in VRAM by the NCCL transfer!")
                break
    else:
        print(f"Notification socket {notify_sock_path} not found. Skipping dynamic update wait.")

if __name__ == "__main__":
    main()
