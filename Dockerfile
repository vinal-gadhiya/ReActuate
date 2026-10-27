# syntax=docker/dockerfile:1
# Build and run on Jetson AGX Thor with JetPack 7.x and NVIDIA Container Toolkit.
ARG BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG ROBOCASA_REF=4f8a2980def75a55dff96b990745b83540425f09
ARG ROBOSUITE_REF=5ce6643f3092639d08f7b0f90ed1c6a84f50552c

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test "${TARGETARCH:-$(dpkg --print-architecture)}" = arm64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential pkg-config \
        python3 python3-dev python3-venv \
        libegl1 libgl1 libopengl0 libglvnd0 libglfw3 \
        libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 \
        libusb-1.0-0-dev libudev-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

# Fetch immutable upstream snapshots during the image build.
RUN mkdir -p /opt/src /opt/build-info \
    && git init /opt/src/robosuite \
    && git -C /opt/src/robosuite remote add origin https://github.com/ARISE-Initiative/robosuite.git \
    && git -C /opt/src/robosuite fetch --depth 1 origin "${ROBOSUITE_REF}" \
    && git -C /opt/src/robosuite checkout --detach FETCH_HEAD \
    && git -C /opt/src/robosuite rev-parse HEAD > /opt/build-info/robosuite.commit \
    && git init /opt/src/robocasa \
    && git -C /opt/src/robocasa remote add origin https://github.com/robocasa/robocasa.git \
    && git -C /opt/src/robocasa fetch --depth 1 origin "${ROBOCASA_REF}" \
    && git -C /opt/src/robocasa checkout --detach FETCH_HEAD \
    && git -C /opt/src/robocasa rev-parse HEAD > /opt/build-info/robocasa.commit

# Simulator-only packaging: remove the two training dependencies, retaining all
# other upstream requirements and normal pip dependency resolution. No simulator
# code is modified. LeRobot/Tianshou workflows are outside this image's scope.
RUN python - <<'PY'
import ast
from pathlib import Path

path = Path('/opt/src/robocasa/setup.py')
source = path.read_text()
tree = ast.parse(source)
requirements = next(
    keyword.value
    for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    and node.func.id == 'setup'
    for keyword in node.keywords if keyword.arg == 'install_requires'
)
original = ast.literal_eval(requirements)
excluded = {'lerobot==0.3.3', 'tianshou==0.4.10'}
assert excluded <= set(original), 'Upstream requirements changed; review simulator dependencies'
filtered = [item for item in original if item not in excluded]
lines = source.splitlines(keepends=True)
start = sum(map(len, lines[:requirements.lineno - 1])) + requirements.col_offset
end = sum(map(len, lines[:requirements.end_lineno - 1])) + requirements.end_col_offset
path.write_text(source[:start] + repr(filtered) + source[end:])
PY

# Ubuntu 24.04 supplies Python 3.12. MuJoCo 3.3.1 has Linux ARM64 wheels.
# Resolve both packages together so robosuite cannot upgrade RoboCasa's MuJoCo.
RUN python -m pip install \
        -e /opt/src/robosuite -e /opt/src/robocasa imageio-ffmpeg \
    && python -m pip check \
    && python -m pip freeze > /opt/build-info/pip-freeze.txt

# Set up macros without importing packages or requiring a GPU during docker build.
# A fresh named volume is seeded with the bundled assets by Docker.
RUN touch /opt/src/robocasa/robocasa/macros_private.py \
    && touch /opt/src/robosuite/robosuite/macros_private.py \
    && mv /opt/src/robocasa/robocasa/models/assets /opt/robocasa-assets \
    && ln -s /opt/robocasa-assets /opt/src/robocasa/robocasa/models/assets

COPY scripts/smoke_test.py /opt/smoke_test.py
WORKDIR /workspace
CMD ["bash"]
