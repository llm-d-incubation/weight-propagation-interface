cat << 'EOF' > test_mem.py
import ctypes
libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
free = ctypes.c_size_t()
total = ctypes.c_size_t()
libcuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
print(f"Free: {free.value / (1024**3):.2f} GB")
print(f"Total: {total.value / (1024**3):.2f} GB")
EOF

NODE_NAME=$1

if [ -z "$NODE_NAME" ]; then
    echo "No node name provided. Checking memory on all GPUs..."
    PODS=$(kubectl get pods -n wpi-system -l app=wpi-driver -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName --no-headers)
    
    echo "$PODS" | while read -r pod node; do
        echo "======================================"
        echo "Node: $node"
        kubectl cp test_mem.py wpi-system/$pod:/test_mem.py -c wpi-driver
        kubectl exec $pod -n wpi-system -c wpi-driver -- python /test_mem.py
    done
else
    echo "Checking memory on node: $NODE_NAME"
    POD=$(kubectl get pods -n wpi-system -l app=wpi-driver --field-selector spec.nodeName=$NODE_NAME -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$POD" ]; then
        echo "Error: No wpi-driver pod found on node $NODE_NAME"
        exit 1
    fi
    
    kubectl cp test_mem.py wpi-system/$POD:/test_mem.py -c wpi-driver
    kubectl exec $POD -n wpi-system -c wpi-driver -- python /test_mem.py
fi