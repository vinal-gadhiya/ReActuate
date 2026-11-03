# ReActuate shared contracts

Keep shared code independent of MuJoCo, RoboCasa, PyTorch, model SDKs, and
provider-specific types. It must not import `sim` or `inference`; those components
may depend on `core`. Avoid a global dependency set or framework before needed.

`policy_http.py` implements a versioned NumPy transport with no torch or pickle
dependency. The client retrieves JSON metadata and exchanges NPZ observation and
action dictionaries using HTTP. NPZ reads disable pickle. Inference responses
echo request IDs so the simulator can reject stale results. Execution is
synchronous at policy chunk boundaries; the plan and verification schemas below
are implemented by the shared controller contracts.

`agent.py` defines and validates subtask plans, verification decisions, and the
literal-call command language. `SupervisorClient` exchanges visual observations
and agent context with the Meta supervisor. No generated code receives the raw
simulator object or a general-purpose Python execution capability.

`execution.py` validates evidence-backed interruptions of an active stage. The
CLI's episode execution policy keeps physical and supervision budgets separate:
short retry limits do not terminate a valid unfinished stage. Completion and
necessary interventions remain dated model claims, not simulator truth.

`atomic_language.py` validates agent policy instructions against the versioned
atomic descriptions and the catalog's explicitly labelled derived instruction
forms. It accepts instantiated documentation placeholders and split transfer
clauses without rewriting accepted text. The agent runner validates plans,
replans and commands and records form provenance in tool traces. Composite skill
keys guide retrieval of atomic references; they are not compulsory action order
or evidence of checkpoint competence.

`simulator_context.py` supplies the public simulator guide, tool descriptions,
task-category references, and optional operator notes. It never queries hidden
objects or task predicates. The simulator adds current robot measurements and
remaining execution budgets to each request.

`atomic_task_catalog.json` contains all 65 official atomic-task description
templates from the pinned RoboCasa source, with task categories, source paths,
provenance, and the upstream MIT notice. These are public documentation templates,
not exact per-demonstration training annotations or promises of policy competence.
The context loader checks coverage against the installed task registry.

`scene_memory.py` defines structured transfer assessments and bounded semantic
object/fixture memory. The simulator owns a fresh instance per episode and keeps
it across replans. Entries are time-stamped visual claims, not ground-truth poses.
Navigation requires fixture evidence and an explicit information need.

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
