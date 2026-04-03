import mmap
import os
import time
import logging
import socket
import array
import ctypes
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

SOCKET_DIR = "/run/wpi/sockets"

def receive_fd(sock):
    """Receives a file descriptor over a UNIX socket using SCM_RIGHTS."""
    fds = array.array("i", [0])
    msg, ancdata, flags, addr = sock.recvmsg(1, socket.CMSG_LEN(fds.itemsize))
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            return fds[1]
    return None

def get_socket_path():
    if os.path.exists(SOCKET_DIR):
        socks = [f for f in os.listdir(SOCKET_DIR) if f.endswith(".sock")]
        if socks:
            return os.path.join(SOCKET_DIR, socks[0])
    return None

def consume_weights():
    logger.info(f"Waiting for UNIX socket in {SOCKET_DIR} to be created by the driver...")
    sock_path = None
    while not sock_path:
        sock_path = get_socket_path()
        if not sock_path:
            time.sleep(2)
    
    logger.info(f"Socket {sock_path} found. Connecting...")
    
    try:
        import torch

        # Connect to the UNIX socket
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(sock_path)
        
        logger.info("Connected. Waiting to receive file descriptor...")
        fd = receive_fd(client)
        client.close()
        
        if fd is None:
            logger.error("Failed to receive file descriptor.")
            return

        logger.info(f"Successfully received file descriptor: {fd}. Attempting to import via CUDA...")

        try:
            libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
            if libcuda.cuInit(0) != 0: raise Exception("cuInit failed")
            device = ctypes.c_int()
            if libcuda.cuDeviceGet(ctypes.byref(device), 0) != 0: raise Exception("cuDeviceGet failed")
            ctx = ctypes.c_void_p()
            if libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device) != 0: raise Exception("cuCtxCreate failed")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return

        try:
            size = 10 * 1024 * 1024 * 1024 # 10 GiB

            handle = ctypes.c_ulonglong()
            err = libcuda.cuMemImportFromShareableHandle(ctypes.byref(handle), ctypes.c_void_p(fd), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
            if err != 0: raise Exception(f"cuMemImportFromShareableHandle failed with err {err}")
            logger.info(f"Successfully imported shareable handle! Generic Handle: {handle.value}")

            device_ptr = ctypes.c_ulonglong()
            err = libcuda.cuMemAddressReserve(ctypes.byref(device_ptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
            if err != 0: raise Exception("cuMemAddressReserve failed")
            
            err = libcuda.cuMemMap(device_ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0))
            if err != 0: raise Exception("cuMemMap failed")

            desc = CUmemAccessDesc()
            desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            desc.location.id = device.value
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

            err = libcuda.cuMemSetAccess(device_ptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))
            if err != 0: raise Exception("cuMemSetAccess failed")

            logger.info("Successfully mapped and set access for GPU memory! VRAM Sharing via WPI is verified!")

            # Use PyTorch __cuda_array_interface__ for zero-copy VRAM mapping
            class RawCUDATensor:
                def __init__(self, ptr, nbytes):
                    self.__cuda_array_interface__ = {
                        "shape": (nbytes,),
                        "typestr": "|u1",
                        "data": (ptr, False),
                        "version": 3,
                    }

            logger.info("Wrapping device pointer with PyTorch...")
            raw_array = RawCUDATensor(device_ptr.value, size)
            tensor = torch.as_tensor(raw_array, device=torch.device('cuda:0'))
            
            # Cast to float16
            weights = tensor.view(torch.float16)

            logger.info(f"Successfully wrapped GPU memory into PyTorch Tensor! Shape: {weights.shape}")
            logger.info(f"First 10 elements: {weights[:10]}")
            logger.info(f"Chunk mean: {weights.float().mean().item()}")

            # Wait keeping the pod alive
            logger.info("Weight consumption successful. Keeping pod alive...")
            while True:
                time.sleep(60)
                
        except Exception as e:
            import traceback
            logger.error(f"Error during CUDA import/map phase: {e}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error connecting or executing consumer: {e}")

if __name__ == "__main__":
    consume_weights()
