# ReActuate inference and supervision

Own VLA serving, model-specific observation/action adapters, Meta API clients,
visual evidence extraction, planning, progress verification, and recovery.
Orchestration belongs here; `core/` holds only shared definitions.

The supervisor uses a separate lightweight CPU image and calls the hosted Meta
API for Muse and optional SAM. Frozen-policy inference runs on Thor's GPU.
Select one backend using the corresponding Compose overlay:

| Backend | Compose files | Setup |
| --- | --- | --- |
| GR00T N1.5 | `compose.yaml` | [GR00T run guide](RUNNING.md) |
| PatchWAM | `compose.yaml` + `compose.patchwam.yaml` | [PatchWAM guide](PATCHWAM.md) |
| Awomo-0.5I | `compose.yaml` + `compose.awomo.yaml` | [Awomo guide](AWOMO.md) |

Awomo agent runs on this branch default to [situational control](SITUATIONAL.md).
Before execution, Muse describes the task's required final state, surveys the
scene, identifies uncertainty, and proposes a plan. The same call also chooses
the first instruction. Later calls receive persistent planning state and revise
the plan, scene facts and progress claims. Muse can change atomic-language
instructions each feedback interval. There are no independent verifiers or
subtask boundaries in this mode. SAM remains optional. The simulator determines
task success and termination; the physical horizon and explicit operational
limits remain in force.

Use `--agent-control staged` for Awomo's parent controller. The
[staged agent guide](AGENT.md) covers planning windows, bounded programs, visual
verification, recovery declarations and matched-scene comparisons. GR00T and
PatchWAM default to staged control. Baseline runs use the original task instruction.

Implementation entry points:

- `serve.py`: serialized frozen-policy HTTP serving and backend selection.
- `serve_awomo.py`: official Awomo policy integration and instruction updates.
- `supervisor.py`: Muse action selection, staged supervision and optional SAM.
- `meta_api.py`: hosted API requests, structured outputs and bounded output retries.
- `perception.py`: SAM image/clip evidence adapters.
- `Dockerfile*` and `download_*.py`: backend images and checkpoint acquisition.

Shared schemas, planning records and transport clients live in `core/`; the
simulator owns physical execution and evaluation in `sim/`.

The default GR00T backend targets the RoboCasa N1.5 fork at
`9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`, with the linked multitask
`checkpoint-120000`. Its metadata, normalization, and policy transforms are
preserved. Follow [RUNNING.md](RUNNING.md) for container builds, checkpoint
download, startup, tests, and single-task evaluation.

An optional [PatchWAM backend](PATCHWAM.md) has its own pinned source, image,
asset downloader, and Compose overlay. It uses the same simulator/supervisor
protocol without changing the default GR00T backend.

`serve.py` exposes `/health`, `/metadata`, and `/predict` over HTTP. It uses the
upstream `Gr00tPolicy` with the `panda_omron` data config. The upstream transport
requires torch serialization in its client; the NumPy transport in `core/`
keeps that dependency out of the simulator. The server serializes policy calls
and returns the originating request ID alongside each action chunk.

`Dockerfile` installs the pinned inference requirements and official CUDA 13
PyTorch ARM64 wheels. `prepare_upstream.py` selects the inference dependency list
and eager attention backend; it does not alter checkpoint weights or transforms.
`download_checkpoint.py` downloads only inference artifacts and records their
resolved revision. End-to-end GPU execution requires validation on Thor.

Configure provider endpoint, model identifier, timeouts, and budgets. Muse Spark
on Meta Model API and Muse Spark Open through a hosted provider need separate
adapters. Verify account access, image inputs, structured outputs, and rate
limits before committing to a model. Supply credentials through runtime secrets
or environment variables, never committed files.

Maintain dated visual evidence across decisions. API/decision caps default to
unlimited; retries remain bounded, and explicit caps are available for experiments.
Situational failures never authorize replaying a stale instruction; staged
fallback rules are documented separately. Discard pending action tails when an
instruction changes. Reset episode state between episodes; Awomo retains its
valid within-episode visual history across instruction changes.

Images and default checkpoint/cache volumes use the `reactuate-` prefix. See the
[root README](../README.md#existing-docker-installations) for volume overrides
that reuse existing downloads after renaming. Tests, builds and GPU checks run
in Thor containers; mocked protocol tests do not establish live task quality.
