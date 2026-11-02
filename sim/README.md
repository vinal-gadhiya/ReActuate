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
- `batch.py`: sequential overnight queues, with per-experiment logs, wall-clock
  timing, optional timeouts (disabled by default), and summaries updated after
  each experiment. See [overnight runs](OVERNIGHT.md).

Agent episodes retain semantic object memory across decisions and replans. Muse
can request perception without motion, select a camera/target for SAM detection
or clip tracking, and request bounded fixture navigation before resuming a pending
transfer. Motion diagnostics compare declared robot measurements and rendered
images; they do not query private object poses or certify task progress.

Supervision defaults to `--verification-mode selective`: one planner call for
ordinary continuation, followed by an independent check for completion, recovery,
new acquisition, loss/identity changes or consequential requirement/attempt claims.
Both calls retain all supplied RGB cameras, dated history and optional depth.
Rejected consequential claims produce inspection and feedback without applying
their actions or memory/goal changes. `--verification-mode always` retains the
previous pre-action review schedule for ablation. See [latency evaluation](LATENCY.md)
for Thor commands, instrumentation and the performance acceptance criteria.

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
work first uses a complete source-to-destination transfer instruction, including
transport within the same policy request. Completion requires final placement,
not just pickup or arrival. Observed failure or a necessary verified recovery can
decompose the pending transfer into pick, navigate and place, retaining the original
goal/destination and omitting phases already achieved. Access/opening
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
Decision/API limits cap supervision; set `--max-decisions 0 --max-api-calls 0`
to keep supervision active without total decision or API-call caps. Set
`--max-replans -1` to remove the recovery replan cap. The overnight example uses
these options and a 4,000-step physical budget, with no wall-clock deadline.
With finite decision/API limits, when exhausted, bounded policy bursts keep
the unfinished committed stage without advancing unverified work. Invalid
proposals and model stops also trigger logged continuation. A completed window
still requires supervision to supply a valid next window; no new action is invented
if that cannot be obtained. Real execution errors remain errors.

In episode mode, repeated inspection at a window boundary requests more planning
instead of ending with `perception_stall`. A matching current verifier dispute
can reopen the last completed stage, including at the same simulator step.
Related satisfaction claims become unknown, with the old claims retained in
audit history; this does not assert object loss or physical failure. The planner
resumes the original pending instruction or proposes a justified recovery.
Placement verification separately checks destination support and gripper release;
an object still carried toward its destination is partial progress. Completion
review omits the planner's assessment narrative to reduce agreement bias.

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
