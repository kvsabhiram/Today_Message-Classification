# ToDoZee Task Classifier — GPU inference image (Qwen2.5-3B + LoRA v11)
# Target hardware: AWS g5.xlarge (NVIDIA A10G, Ampere) — native bf16, CUDA 12.6.
# The torch+cu126 wheels bundle the CUDA libs; the cudnn-runtime base supplies cuDNN.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1

# Python 3.10 (matches .python-version) + git/git-lfs for any LFS-tracked weights + curl for healthcheck.
# gcc/build-essential are REQUIRED: peft loads the LoRA adapter -> imports bitsandbytes -> triton,
# which JIT-compiles a CUDA helper at model-load time and needs a C compiler. The cuda-runtime base
# has none, so without this the server crashes on startup ("Failed to find C compiler").
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git git-lfs curl ca-certificates \
        gcc build-essential \
    && git lfs install \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
# requirements.txt carries its own --extra-index-url for the cu126 wheels.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Bake the base model into the image so the container has no runtime dependency on
# the HuggingFace Hub (faster, reproducible cold-starts). ~6GB.
ARG BASE_MODEL=Qwen/Qwen2.5-3B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${BASE_MODEL}')"

# Application code + the merged LoRA adapter (output_v11 incl. adapter_model.safetensors via LFS).
COPY inference.py .
COPY output_v11 ./output_v11

ENV ADAPTER_PATH=/app/output_v11 \
    BASE_MODEL=Qwen/Qwen2.5-3B \
    HOST=0.0.0.0 \
    PORT=5011

EXPOSE 5011

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:5011/health || exit 1

CMD ["python", "inference.py"]
