# ReActuate

ReActuate explores how Meta model APIs can improve a frozen vision-language-action
(VLA) policy through planning, visual verification, and recovery in RoboCasa.

```text
sim/         RoboCasa execution, rendering, rollouts, evaluation, simulator image
inference/   VLA serving, Meta API adapters, planning, perception, recovery
core/        Shared interfaces, configuration conventions, trace formats
```

Simulation, frozen-policy inference and Muse supervision run in separate
containers on Jetson AGX Thor. MuJoCo physics uses the CPU; policy inference
and EGL rendering use the GPU. The supervisor calls hosted Meta model APIs.

## Setup and evaluation

This README describes branch `main` and its existing controller.

- [GR00T N1.5 setup and atomic-task evaluation](inference/RUNNING.md)
  uses `compose.yaml` and the released multitask checkpoint.
- [PatchWAM setup](inference/PATCHWAM.md) uses `compose.yaml` plus
  `compose.patchwam.yaml` and a separate checkpoint volume.
- [Muse agent execution and optional SAM](inference/AGENT.md) covers this
  branch's planning, verification, recovery and matched-scene comparisons.

Select one policy overlay at a time. Baseline runs use the original task
instruction. Run builds, Python tests, simulation and inference in Thor
containers. This branch defaults to 40 supervisor decisions and 60 API calls per episode.
Use explicit budgets appropriate to the run; consult this branch's CLI help.
The simulator's task check determines whole-task success. Outputs include
videos, labelled observations, traces, API audit and task results; see the
backend/agent guides for commands and the controller behavior on this branch.

## Experiment branches

Choose the branch for the controller you want to evaluate, then follow that
branch's README and setup guides. Features and CLI options differ between
branches; the Awomo experiments are available on the branches listed below.

| Branch | Purpose | Documentation |
| --- | --- | --- |
| `main` | GR00T N1.5 and PatchWAM with Muse planning, verification and recovery. | This README |
| `experiment/awomo-situational` | Awomo with flexible Muse instructions, initial task/scene planning and persistent progress tracking, without independent verifiers or subtask boundaries. | [Branch README](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2Fawomo-situational/README.md), [Awomo setup](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2Fawomo-situational/inference/AWOMO.md), [situational runs](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2Fawomo-situational/inference/SITUATIONAL.md) |
| `experiment/2953296-full-access-transfer` | Awomo, GR00T and PatchWAM with staged supervision, paired-door checks, placement/release verification and optional Awomo relational instructions. | [Branch README](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2F2953296-full-access-transfer/README.md) |
| `experiment/2953296-evaluation` | Earlier GR00T/PatchWAM experiment with atomic instruction grounding and stage boundary confirmation. | [Branch README](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2F2953296-evaluation/README.md) |
| `experiment/244b087-simulator-termination` | Earlier experiment adjusting termination behavior and instruction/recovery flexibility. | [Branch README](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2F244b087-simulator-termination/README.md) |
| `baseline/244b087-evaluation` | Historical harness for comparisons with the earlier controller. | [Branch README](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/baseline%2F244b087-evaluation/README.md) |

For the flexible Awomo experiment, switch branches on Thor before building:

```bash
git fetch origin
git switch experiment/awomo-situational
git pull --ff-only
```

Then follow that branch's [Awomo setup](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2Fawomo-situational/inference/AWOMO.md)
and [situational run commands](https://github.com/vinal-gadhiya/meta-ai-hackathon2026/blob/experiment%2Fawomo-situational/inference/SITUATIONAL.md).
Rebuild/recreate the services as directed there so the running containers use
the selected branch's code. These are experiment variants; their names do not
establish a ranking in task success or latency. Use matched scenes/seeds when
comparing them.

## Existing Docker installations

ReActuate Docker images and default checkpoint/cache volumes use the
`reactuate-` prefix. Rebuild the selected images after updating. The simulator
image remains `robocasa-thor:latest`, and kitchen assets retain the
`robocasa-thor-assets` volume.

To reuse prior downloads, find the existing names with `docker volume ls`
and set the appropriate variables in your local `.env` before invoking Compose:

| Existing data | Volume-name override |
| --- | --- |
| GR00T checkpoints | `REACTUATE_CHECKPOINT_VOLUME` |
| PatchWAM checkpoints | `REACTUATE_PATCHWAM_CHECKPOINT_VOLUME` |
| Hugging Face cache | `REACTUATE_HF_CACHE_VOLUME` |

Overrides reuse existing storage; the rename does not copy or delete volumes.
Without overrides, downloads must populate the newly named volumes.

## RoboCasa and MuJoCo on Jetson AGX Thor

This image targets **Linux ARM64 on Jetson AGX Thor with JetPack 7.x**,
NVIDIA's host graphics driver, Docker and NVIDIA Container Toolkit. It uses
CUDA 13.0 / Ubuntu 24.04, Python 3.12, pip, MuJoCo 3.3.1, and pinned Git
snapshots of RoboCasa365 1.0.1 and robosuite 1.5.2. It defaults to headless EGL
camera rendering; no display server is needed.

Thor uses the SBSA software stack, which supports standard ARM64 CUDA
containers. See [NVIDIA's Thor container instructions](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html).
The host must have a driver compatible with CUDA 13.0. The image does not install
or replace the host driver. `BASE_IMAGE` can be overridden if your host requires
a different compatible CUDA 13 / Ubuntu 24.04 image.

## Build on Thor

Copy this repository to the Thor and run from its root:

```bash
docker build --platform linux/arm64 -f sim/Dockerfile -t robocasa-thor:latest .
```

Builds download upstream source and Python dependencies. No GPU is needed during
the build. Build on Thor to avoid ARM emulation on another architecture.

## Start the container

```bash
mkdir -p outputs
docker run --rm -it \
  --runtime=nvidia --gpus all \
  --shm-size=2g \
  -v robocasa-thor-assets:/opt/robocasa-assets \
  -v "$(pwd)/outputs:/outputs" \
  robocasa-thor:latest
```

Inside the container, first check MuJoCo physics and actual NVIDIA EGL rendering:

```bash
python /opt/smoke_test.py --output /outputs
```

Then download kitchen assets and run the complete kitchen check:

```bash
python -m robocasa.scripts.download_kitchen_assets
python /opt/smoke_test.py --kitchen --output /outputs
```

Answer `y` to the asset prompt. The upstream downloader estimates about 10 GB;
allow additional disk space for extraction. Inspect its output for failures: it
can report failed downloads without returning a nonzero exit status. Assets
persist in the named volume across container removal. A fresh named volume is
seeded with the assets bundled in the image; an empty host bind mount at this
path would hide those bundled files. Use a fresh volume for a different source
revision unless you have checked asset compatibility.

Successful checks write `mujoco.png` and `robocasa.png` into the host's `outputs`
directory. The kitchen check uses PandaOmron, resets a task, takes ten simulation
steps, and checks a rendered camera frame. First use may take longer for Numba
compilation. Files in the output mount are created as root.

## Simulator scope and dependency choices

RoboCasa's [upstream package requirements](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/setup.py)
include LeRobot and Tianshou. During the image build, their two entries are
removed from the local packaging metadata so simulator installation does not
pull in training frameworks, PyTorch or TorchCodec. All remaining upstream
requirements are installed with normal dependency resolution and `pip check`.
Simulator source code is unchanged. Training, LeRobot evaluation, and optional
humanoid IK integrations are outside this image's scope.

MuJoCo's standard physics engine runs on the CPU; EGL camera rendering uses
Thor's NVIDIA GPU. This image does not enable MJX or GPU physics.

The NVIDIA runtime supplies the host's driver libraries at container startup.
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` includes the graphics
libraries required for OpenGL/EGL; see [NVIDIA's capability documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
Do not install an NVIDIA display driver inside this image. An EGL failure must
be resolved before trying the kitchen check; `nvidia-smi` alone does not verify
rendering. The smoke test prints the OpenGL vendor and renderer and rejects a
non-NVIDIA renderer.

Sources live under `/opt/src`, and the Python environment is under `/opt/venv`,
so you can mount project code over `/workspace` without hiding the installation.
Source revisions and resolved Python versions are recorded in `/opt/build-info`.
Source commits are pinned; transitive dependencies, base image digest, and apt
packages are not fully locked. Override `ROBOCASA_REF` and `ROBOSUITE_REF` as
build arguments to deliberately change source snapshots.
