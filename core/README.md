# Shared contracts

Keep shared code independent of MuJoCo, RoboCasa, PyTorch, model SDKs, and
provider-specific types. It must not import `sim` or `inference`; those components
may depend on `core`. Avoid a global dependency set or framework before needed.

`policy_http.py` implements a versioned NumPy transport with no torch or pickle
dependency. The client retrieves JSON metadata and exchanges NPZ observation and
action dictionaries using HTTP. NPZ reads disable pickle. Inference responses
echo request IDs so the simulator can reject stale results. This initial runner
is synchronous; broader skill/verification schemas below remain planned.

`agent.py` defines and validates subtask plans, verification decisions, and the
literal-call command language. `SupervisorClient` exchanges visual observations
and agent context with the Meta supervisor. No generated code receives the raw
simulator object or a general-purpose Python execution capability.

| Message | Required meaning |
| --- | --- |
| Observation | Schema version; run/episode/step IDs; simulated timestamp; instruction version; task instruction; RGB frames/camera IDs; proprioception; optional declared depth/calibration |
| Action chunk | Originating observation/instruction IDs; action schema; embodiment/control mode; bounded horizon; ordered actions |
| Skill proposal | Validated skill identifier; grounded arguments; instruction; preconditions; observable completion criteria; step/retry budget |
| Verification | `satisfied`, `unsatisfied`, or `unknown`; supporting observation IDs; evidence; intervention reason |
| Episode trace | Resolved config; source/image/checkpoint revisions; seed; decisions; outcomes; simulator steps; timings; API usage/cost |

Define units, coordinate frames, image orientation, tensor shapes, normalization,
camera order, and action semantics in each model adapter. Reject schema,
embodiment, and episode mismatches before stepping the environment.

Separate privileged evaluation labels from controller observations in messages
and traces. Record model/prompt versions and responses with credentials removed.
Start with append-only JSONL traces; refer to large image/video files by path.
Configuration should select task/seed, execution mode, provider, intervention
schedule, budgets, and output location.
