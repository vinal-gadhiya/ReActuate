# ReActuate inference and supervision

Own VLA serving, model-specific observation/action adapters, Meta API clients,
visual evidence extraction, planning, progress verification, and recovery.
Orchestration belongs here; `core/` holds only shared definitions.

The [Meta agent guide](AGENT.md) describes the implemented Muse planner,
supervisor service, bounded command programs, optional SAM grounding, and
composite-task runs. The supervisor uses a separate lightweight CPU image.

Implemented entry points are `serve.py` for serialized frozen-policy HTTP
serving, `supervisor.py` for Muse planning/decisions and optional SAM,
`meta_api.py` for hosted API transport, and `perception.py` for visual evidence.
Shared schemas and protocol clients live in `core/`; physical execution lives
in `sim/`. No general-purpose generated Python runs against the simulator.

The default GR00T backend targets: the RoboCasa GR00T N1.5 fork at
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

Use API calls at task start and selected boundaries. Maintain visual evidence
over time, allow `unknown` verdicts, and cap calls/retries per episode. Reject
malformed decisions and unsupported skills. On timeout, follow a configured
bounded fallback or end the episode with a recorded reason. Reset VLA temporal
state and discard pending actions when the active instruction changes.

This branch defaults to 40 supervisor decisions and 60 API calls per episode.
Use explicit budgets appropriate to the run; consult this branch's CLI help.
Image names and checkpoint/cache defaults now use the `reactuate-` prefix.
See [existing Docker installations](../README.md#existing-docker-installations)
to reuse prior checkpoint/cache volumes through environment overrides.
Tests, builds and GPU execution checks run in Thor containers.
