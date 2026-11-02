# Simulation

Own RoboCasa/MuJoCo setup, environment lifecycle, rendering, simulator-side
policy clients, rollout recording, task scoring, and experiment execution.

- `Dockerfile`: Thor simulator image. Build from the repository root with
  `docker build -f sim/Dockerfile -t robocasa-thor:latest .`.
- `smoke_test.py`: physics, NVIDIA EGL, and optional kitchen checks.
- `evaluate.py`: atomic-task evaluation against the GR00T HTTP service, with
  per-episode videos and JSON results. See [the run guide](../inference/RUNNING.md).
- `run_agent.py`: atomic or composite tasks with the original instruction, a
  fixed subtask schedule, or Muse-supervised execution. See [the agent guide](../inference/AGENT.md).
- `compare.py`: sequential comparisons of GR00T, Muse, and Muse + SAM on matched
  scene seeds, with videos, traces, and aggregated task-success and latency results.

Agent episodes retain semantic object memory across decisions and replans. Muse
can request perception without motion, select a camera/target for SAM detection
or clip tracking, and request bounded fixture navigation before resuming a pending
transfer. Motion diagnostics compare declared robot measurements and rendered
images; they do not query private object poses or certify task progress.

`--planning-context summary` (default) also supplies a bounded episode notebook
with the initial/current plan, accepted revisions and completion claims, executed
attempts, object location changes, and recovery feedback. It survives the recent
event window and resets between episodes. `episode_context.json` saves the latest
summary; `supervisor_request` trace events save the exact context sent to Muse.
`--planning-context recent` omits this notebook for comparison while keeping object
memory and the existing recent-event/handoff context. No extra API calls or
physical actions are introduced. See [the PatchWAM guide](../inference/PATCHWAM.md#in-context-planning-and-object-memory).

The selected frozen policy (GR00T or PatchWAM) owns all control actions. The CLI no longer offers `--hybrid-control` or
`--allow-gripper`: no action arbitration, forced grasp latch, manual release,
base/orientation freeze, or Cartesian placement corrections are applied. Muse
and SAM provide planning, visual memory, verification, and language corrections.
Destination visibility does not automatically shorten a policy execution burst.
Agent policy instructions use atomic-catalog vocabulary and formats, with exactly
one tool call per decision and observation before the next stage. New pick-and-place
objectives first use a complete source-to-destination transfer instruction;
only observed wrong execution or justified recovery permits decomposition into
pick, navigate and place. Final placement/release is required for completion.
For paired-door fixtures both doors must be open, with per-panel confirmation;
single-door fixtures and drawers do not require invented partners. See
[this branch's experiment notes](FULL_ACCESS_TRANSFER.md). Access/opening
is conditional on an observed obstruction or explicit requirement. Generated
plans, replans and policy requests must match a documented atomic description or
an explicitly derived `instruction_forms` template in the atomic catalog. Entity
names fill the slots; unsupported synonyms and appended motor explanations are
rejected before motion with correction feedback. Accepted text is passed through
unchanged, with no 12-word/96-character limit. Tool traces record matched form
provenance. Baseline text is unchanged. Rebuild both sim and supervisor and
regenerate old plans that contain free-form language to use this contract.
Generated plans include a visual inventory, apparent access, first-action
rationale and route rationale, recorded in `scene_assessment.json` and `plan.json`.
Current survey claims seed memory before the first decision without motion or an
extra model call. Supplied-plan surveys remain historical. Explicit placement
destinations are checked against the active stage before execution.

The CLI defaults to `--execution-policy episode`: an unfinished stage keeps its
committed atomic instruction until verified complete, a necessary evidence-backed
intervention, simulator success/end, or the physical episode horizon. Use
`--max-steps 2000` for a 2,000-step horizon. A burst is an observation interval;
short failure/recovery limits and planned subtask ceilings do not end execution.
Decision/API limits cap supervision; when exhausted, bounded policy bursts keep
the unfinished committed stage without advancing unverified work. Invalid
proposals and model stops also trigger logged continuation. A completed window
still requires supervision to supply a valid next window; no new action is invented
if that cannot be obtained. Real execution errors remain errors.

Changing an unfinished stage requires current observation references and a
specific `recovery.necessity`, not merely a few failed bursts or another tool
alias. Completion needs current satisfied evidence. These rules are shared across
all tasks. See the agent guide for intervention reasons and deployment commands.

With `--execution-policy bounded`, `run_agent.py` and `compare.py` accept `--supervisor-stop-policy continue` for
simulation comparisons that should keep trying within the episode budget. A Muse
`stop` request then retries the last successfully executed `run_groot` instruction
for the current subtask for at most 16 steps, followed by another supervisor
decision on the resulting observation. Each request remains subject to the usual
program validation and decision/API budgets. Navigation
and instructions from prior subtasks are never replayed by this
fallback. The trace records `supervisor_stop_deferred`; results record the chosen
policy. This does not guarantee use of the full horizon or task success. The
default `halt` policy still ends immediately on a model stop request. Agent CLI
continuation is bounded by `--max-stop-continuations` (default 2 consecutive
deferred stops), then returns `supervisor_stop_retry_limit`. Ordinary non-stop
decisions reset the count; API failure continuation has its separate limit.

Transient Muse decision failures (connection errors, timeouts, rate limits, and
selected upstream server errors) also allow short guarded continuation of the
current subtask's last executed policy instruction before retrying with fresh
images. `--max-supervisor-failures 3` permits three consecutive failure bursts;
the fourth failure stops with `supervisor_failure_limit`. A successful response
resets this counter; `0` disables recovery. Each failure consumes a decision and
counts attempted upstream calls, with unknown token usage marked in the trace.
Without a current subtask policy instruction, retries execute no motion.
Authentication, malformed requests,
and unexpected server errors remain errors. Initial planning has no policy
instruction to continue and still fails if its API request fails. Rebuild both
`sim` and `supervisor` and recreate the supervisor service to use the structured
failure protocol.

Assets use `robocasa-thor-assets:/opt/robocasa-assets`; generated artifacts go
under ignored `outputs/`. Keep model weights and API credentials out of this image.

The simulation process owns the authoritative episode, step, and instruction
version. Send observations to inference and validate returned actions before
applying them. Start synchronously at action-chunk boundaries: pause simulated
time during inference/API calls and measure wall-clock latency separately.
Reject responses whose episode, observation, or instruction is obsolete.

Privileged task checks and MuJoCo state belong in evaluation/label collection.
Expose only declared sensors and proprioception to the normal controller.
Use a separately named oracle experiment for privileged controller inputs.

The evaluation client uses the shared NumPy HTTP interface, keeping torch and
the GR00T package out of the simulator image. RoboCasa's Gym wrapper handles the
PandaOmron observation/action mapping; the inference service applies the
checkpoint's own policy transforms. For the upstream random-rollout helper, pass
`env.unwrapped` to avoid Gymnasium's `OrderEnforcing` attribute-forwarding error.
