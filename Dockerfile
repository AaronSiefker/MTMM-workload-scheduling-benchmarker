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
