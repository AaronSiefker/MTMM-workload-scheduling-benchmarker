FROM nvcr.io/nvidia/pytorch:26.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:$PATH

RUN apt-get update && apt-get install -y \
   python3-pip \
   python3-dev \
   libprotobuf-dev protobuf-compiler \
   libopenblas-dev \
   vim \
   git wget curl && \
   rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install onnx onnxruntime onnxscript ultralytics seaborn timm==0.4.12 scipy
RUN pip3 install https://download.pytorch.org/whl/nightly/cu132/torchaudio-2.11.0.dev20260515%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl
RUN pip3 install stable-baselines3 sb3_contrib gymnasium

WORKDIR /workspace

CMD ["/bin/bash"]

# Command:
#docker run --runtime nvidia -it -v "$(pwd):/workspace" --privileged
# -v "<datasets_location>:/root/datasets" -v
# "<scheduler_inputs_location>:/root/scheduler_inputs"
# -v "/usr/bin/jetson_clocks:/usr/bin/jetson_clocks" -v "/sys/kernel/debug:/sys/kernel/debug"
# -v "/lib/aarch64-linux-gnu/libxcb.so.1:/lib/aarch64-linux-gnu/libxcb.so.1"
# -v "/lib/aarch64-linux-gnu/libGL.so.1:/lib/aarch64-linux-gnu/libGL.so1"
# -v "/lib/aarch64-linux-gnu/libXau.so.6:/lib/aarch64-linux-gnu/libXau.so.6"
# -v "/lib/aarch64-linux-gnu/libXdmcp.so.6:/lib/aarch64-linux-gnu/libXdmcp.so.6"
# -v "/lib/aarch64-linux-gnu/libGLdispatch.so.0:/lib/aarch64-linux-gnu/libGLdispatch.so.0"
# -v "/lib/aarch64-linux-gnu/libGLX.so.0:/lib/aarch64-linux-gnu/libGLX.so.0"
# -v "/lib/aarch64-linux-gnu/libX11.so.6:/lib/aarch64-linux-gnu/libX11.so.6"
# -v "/usr/bin/tegrastats:/usr/bin/tegrastats"
# --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm <tag>:latest

# Should be executed in ../MTMM-workload-scheduling-benchmarker
