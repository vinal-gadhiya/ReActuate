# syntax=docker/dockerfile:1
ARG BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG GROOT_REF=9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    GROOT_SOURCE_REF=${GROOT_REF} \
    NO_ALBUMENTATIONS_UPDATE=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test "${TARGETARCH:-$(dpkg --print-architecture)}" = arm64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git git-lfs build-essential cmake pkg-config \
        python3 python3-dev python3-venv libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

# Official CUDA 13 wheels include Linux aarch64 / Python 3.12 builds.
RUN python -m pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && mkdir -p /opt/build-info \
    && python -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("torchvision==" + version("torchvision"))' > /opt/build-info/torch-constraints.txt
ENV PIP_CONSTRAINT=/opt/build-info/torch-constraints.txt

# Skip upstream demo media; hydrate only LFS files used by the runtime package.
RUN git lfs install --system \
    && git init /opt/Isaac-GR00T \
    && git -C /opt/Isaac-GR00T remote add origin https://github.com/robocasa-benchmark/Isaac-GR00T.git \
    && git -C /opt/Isaac-GR00T fetch --depth 1 origin "${GROOT_REF}" \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /opt/Isaac-GR00T checkout --detach FETCH_HEAD \
    && git -C /opt/Isaac-GR00T lfs pull --include="gr00t/**" --exclude="" origin \
    && git -C /opt/Isaac-GR00T rev-parse HEAD > /opt/build-info/gr00t.commit

COPY inference/requirements.txt inference/prepare_upstream.py /opt/setup/
RUN python /opt/setup/prepare_upstream.py /opt/Isaac-GR00T /opt/setup/requirements.txt \
    && python -m pip install --only-binary=av -e /opt/Isaac-GR00T

COPY inference/repair_cuda_metadata.py /opt/setup/
RUN python /opt/setup/repair_cuda_metadata.py && python -m pip check
RUN python -c 'from gr00t.model.policy import Gr00tPolicy; from gr00t.experiment.data_config import DATA_CONFIG_MAP; assert "panda_omron" in DATA_CONFIG_MAP' \
    && python -m pip freeze > /opt/build-info/pip-freeze.txt

COPY core /app/core
COPY inference /app/inference
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --start-period=300s --interval=10s --timeout=5s --retries=3 \
    CMD python -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8000/health", timeout=3)'
CMD ["python", "-m", "inference.serve", "--checkpoint", "/checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"]
