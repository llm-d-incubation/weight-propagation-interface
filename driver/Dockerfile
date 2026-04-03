# 1. Build the Go DRA Plugin
FROM golang:1.25 AS go-builder
WORKDIR /go/src/app
COPY operator operator
COPY driver/dra-plugin dra-plugin
WORKDIR /go/src/app/dra-plugin
RUN sed -i 's|../../operator|../operator|g' go.mod
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o /dra-plugin .

# 2. Build the Python WPI Driver
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libnl-route-3-200 \
    libnl-3-200 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    numpy \
    cupy-cuda12x \
    nvidia-nccl-cu12 \
    torch \
    safetensors

COPY proto/wpi.proto ./proto/
COPY driver/main.py .

# Generate gRPC bindings
RUN python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/wpi.proto

# Copy the Go plugin binary
COPY --from=go-builder /dra-plugin /usr/local/bin/dra-plugin

ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}
EXPOSE 50051

CMD ["python", "main.py"]
