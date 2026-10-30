# Meta-guided composite-task execution

The agent decomposes a RoboCasa goal into short instructions, supervises each
subtask using camera observations, and generates bounded programs over two tools:

```python
run_groot(instruction="Open the cabinet door beside the sink.", steps=40)
set_gripper(closed=False, steps=5)
```

Programs are parsed into validated literal calls, never executed with Python
`eval` or `exec`. Gripper control is opt-in and available only for the expected
PandaOmron action schema with a fixed-impedance, symmetric delta OSC controller.
Cartesian motion planning, inverse kinematics, and automatic skill-library
learning are not implemented in this first version.

## What context the model receives

Each planning and verification request includes a public simulator guide:

- PandaOmron's wheeled base, single arm, gripper, and fallible GR00T executor.
- The current task name/type and original episode instruction.
- Camera identities, resolutions, current RGB images, and robot proprioception.
- Enabled tools and their limits, including whether direct gripper control is available.
- Registered atomic task names as reference vocabulary, not callable tools or
  claims that GR00T has mastered those tasks.
- Episode, decision, and API budgets, with an optimistic bound on remaining
  executable motion steps. Subtask budgets must account for these limits.
- Explicit rules about occlusion, distractors, required prerequisites, release,
  observable completion, and persistent scene state.

The initial planning request receives the same capability context as later
decisions. `simulator_context.json` records the static context for each episode;
`trace.jsonl` records the actual context and robot measurements sent per request.

The model does not receive a complete simulator object inventory, hidden object
poses, or individual live task predicate results. It receives public task success
requirements where catalogued and the overall simulator success flag. Objects named by the task can be hidden; objects seen
in images can be distractors. Prompts require the model to distinguish these
sources of evidence rather than inventing locations or treating occlusion as
absence. This reduces unsupported plans but does not guarantee feasibility.

Optional operator-written notes can add documented task constraints or scene
observations. Save a JSON file under the mounted `outputs/` directory:

```json
{
  "task_notes": "After placing an object, release it and move the gripper clear.",
  "scene_notes": "",
  "constraints": ["Do not add cleanup steps that the original goal does not request."]
}
```

Add `--context-file /outputs/task-context.json` to `sim.run_agent`. Notes are
attributed to the operator, retained in run artifacts, and not treated as
independently verified observations. Keep hidden simulator annotations out of
normal runs; disclose any oracle-assisted conditions separately. No extra API
call is needed to supply this context.

## Atomic instruction grounding

The supervisor supplies all 65 official RoboCasa atomic-task description templates
to Muse on every planning and decision request. The snapshot is stored in
`core/atomic_task_catalog.json`, sourced from
[the pinned description table](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/docs/atomic_tasks/atomic_task_attributes.js)
and [task index](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/docs/atomic_tasks/atomic_task_index.js).
All 65 names were checked against that revision's atomic dataset registry. The
simulator also verifies that its registered atomic tasks have catalog entries,
and saves the catalog reference in `simulator_context.json`. The supervisor
metadata records the catalog revision, task count, and grounding strategy.

Muse now uses concise single-action stages across all agent tasks. The catalog
provides reference vocabulary rather than requiring full-transfer sentences.

Generated plans now require `scene_assessment` before `steps` in the planning
response: a task-relevant inventory with camera references, visibility, location
and apparent access, ordering constraints, the first-action rationale, and route
rationale. Muse surveys all current views and compares accessible targets with
targets needing approach; original task mention order is not an execution order.
The survey and plan share one API call and execute no physical scan. Visual
access is an attributed hypothesis, not metric reachability or optimal routing.
Current survey entries seed semantic memory before the first decision. The full
survey is saved in `scene_assessment.json` and `plan.json`, sent on subsequent
requests with its source, and summarized in the episode notebook. Saved plan
surveys are labeled historical/unverified and never seeded as fresh observations.
Legacy operator plans containing only `steps` remain accepted.

Generated survey camera references are constrained to the actual current labels
in the request schema. Unambiguous short aliases are normalized to those labels;
unknown, ambiguous, and `before:` references are rejected. Invalid model plans
return validation feedback with API usage preserved and receive up to three
budgeted attempts before any physical execution. Healthy planning still uses
one API call. The trace records `invalid_plan` and the next request's correction
feedback if the first proposal needs repair.

Before first motion Muse should correct an obviously unsuitable initial order.
After confirmed placement it should chain eligible work already at the destination,
preserving actual dependencies and final fixture requirements. If a useful object
has already been acquired for another goal, replan to finish its required placement
instead of insisting on the original first object or a temporary setdown.
The runner additionally rejects placement commands that change an explicitly
planned support/receptacle or containment relation. This checks recognizable
`place/put ... on/in/...` text against the active stage, not simulator geometry
or the correctness of a model-generated plan. Exact names and essential modifiers
are preserved; this is not a general synonym resolver or automatic plan repair.

For example: `Pick up the cup.` followed, after confirmed acquisition, by
`Place the cup on the tray.` Navigation, opening, closing, pouring, washing and
other task-relevant actions are separate stages when needed. Keep an instruction
stable while that stage makes progress; do not alternate unrelated goals or
paraphrase merely to justify another retry. Detailed appearance, locations and
verification criteria stay in scene memory, evidence and success criteria.

The agent runner enforces one tool call per decision, at most 12 words and 96
characters for a policy instruction, and rejects combined pickup/place commands
without executing them. It does not automatically rewrite model text. These are
policy language requests, not manual pickup primitives or proof of checkpoint
competence. GR00T and PatchWAM inference code and original-instruction baseline
execution are unchanged.

Access is conditional: an object on an exposed shelf does not require opening a
cabinet. Muse should insert access work only for an observed obstruction or an
explicit task requirement. Zero executed steps in a planned opening stage is not
evidence that opening is necessary. Observe the relevant component before accepting
closure. A supplied combined-transfer plan can still be executed using separate
pickup and placement calls within its active goal, without replanning each stage.

These public templates match the documentation table. They are not a fetched
copy of every language annotation in the checkpoint's training datasets. Exact
episode instructions may name sampled objects or have different wording.
Regenerate plans after rebuilding both `supervisor` and `sim` to use this strategy;
old plans are still accepted but retain their original instruction text.

Instruction stability applies while progress is observed. If stalled, Muse can
temporarily choose the catalog task closest to the next required goal, inspect
its effect, and then resume the pending transfer. For example, `NavigateKitchen`
uses `Navigate to the [kitchen location].` and targets a fixture, not an arbitrary
object or unrestricted exploration. A destination fixture must be grounded in
the task or observations; an unseen glass's location cannot be invented. There
is no generic object-search, exploration, or camera-scan task in the catalog.
Temporary recovery does not mark the original transfer complete, and policy
execution of navigation from a composite-task state remains unverified. Remaining
budget alone is not a reason to repeat a stalled instruction.

## Containers

- `sim`: owns the scene, task score, subtask execution budgets, and video recording.
- `inference`: serves frozen GR00T with the checkpoint's own transforms.
- `supervisor`: a lightweight CPU container calling Meta Model API; no GPU or
  model weights. API keys stay in this container.

Run all commands on Thor from the repository root. No host Python environment
or additional host packages are needed.

## 1. Configure the API

Create `.env` in the repository root on Thor, using a text editor:

```dotenv
MODEL_API_KEY=your_meta_model_api_key
META_MODEL=muse-spark-1.3
META_USE_SAM=0
```

`.env` is ignored by Git and excluded from Docker builds. Compose reads it
directly, including when invoked with `sudo`. Do not commit it, paste the key
into source, or share resolved Compose configuration that contains the key.
To use Wonder, change the model to `muse-wonder-1.0`.

## 2. Build and test

```bash
sudo docker compose --profile agent build supervisor sim
sudo docker compose --profile agent run --rm --no-deps supervisor \
  python -m unittest discover -s tests -v
```

Tests use fake policies/providers. They check instruction replacement, command
validation, subtask transitions, budgets, and transport without API calls or
simulator execution. Existing protocol/packaging tests run here too. Reuse the
working GR00T image, downloaded checkpoint, and simulator asset volume.

## 3. Start services

```bash
sudo docker compose --profile agent up -d --wait inference supervisor
sudo docker compose --profile agent logs -f supervisor
```

`Supervisor ready` means the key is configured and the HTTP service is running;
the first plan request validates account access and the selected model. GR00T
must also report `Policy ready`. Ctrl+C exits log following without stopping the
services. The supervisor API is internal to the Compose network.

## 4. Generate a plan before executing motion

```bash
sudo docker compose --profile agent run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode agent --plan-only \
    --seed 0 --max-api-calls 2 \
    --output /outputs/lemonade-plan-001
```

This creates and resets the scene, sends its RGB views and original instruction
to Muse, and saves `outputs/lemonade-plan-001/episode-000/plan.json`. It does not
step the robot. Inspect the short instructions and observable completion
criteria before a full run. A generated plan is a hypothesis about executable
skills, not proof that GR00T can perform them.

## 5. Execute the supervised sequence

```bash
sudo docker compose --profile agent run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode agent --seed 0 \
    --action-steps 8 --decision-steps 40 \
    --max-api-calls 30 --max-decisions 25 --max-replans 2 \
    --output /outputs/lemonade-agent-001
```

Use a new output directory for each run. To start from the inspected plan, add:

```text
--plan /outputs/lemonade-plan-001/episode-000/plan.json
```

The initial plan can also be written by hand using the same schema:

```json
{
  "steps": [
    {
      "id": "open",
      "instruction": "Open the cabinet door beside the sink.",
      "target": "cabinet door",
      "success_criteria": "The named cabinet door is open enough to access its interior.",
      "max_steps": 300
    }
  ]
}
```

This example illustrates the schema; tailor every instruction to the actual
scene and full task. The agent may complete a subtask, execute another short
program, replace the remaining plan, or stop. Completed steps and recent
execution evidence remain in its context. Plan changes never reset the scene.
Each GR00T call uses the latest observation and temporary subtask instruction;
the original task instruction is retained separately. Pending chunk actions are
discarded at every tool boundary. The current GR00T adapter uses only the current
observation, so it has no cross-subtask temporal history to reset.

GR00T owns all arm, base, orientation, and gripper actions. Muse/SAM plan,
remember visual observations, detect execution mistakes, and change language
instructions. Direct gripper and vertical tools are disabled. The former
`--allow-gripper` and `--hybrid-control` CLI flags have been removed.

## 6. Compare against controlled baselines

Original instruction for the same composite task:

```bash
sudo docker compose run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode baseline --seed 0 --action-steps 8 \
    --output /outputs/lemonade-baseline-001
```

A fixed sequence runs each instruction for its specified budget, without Muse
verification. It is an unverified schedule, useful for isolating decomposition:

```bash
sudo docker compose run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode fixed --seed 0 --action-steps 8 \
    --plan /outputs/lemonade-plan-001/episode-000/plan.json \
    --output /outputs/lemonade-fixed-001
```

Match seeds, task splits, action-chunk lengths, and total simulator-step budgets.
Change `--max-steps` identically across conditions if overriding the task's normal
horizon. Fixed plans should be chosen on development scenes before held-out
evaluation. Neither baseline nor fixed mode calls Meta APIs.

## Optional SAM grounding

Use `--sam` or `--no-sam` on `sim.run_agent` to choose grounding per run without
recreating the supervisor. If neither is supplied, the supervisor's environment
setting is used. These flags affect SAM calls at decision boundaries, not GR00T.

Set `META_USE_SAM=1` in `.env`, then recreate only the supervisor:

```bash
sudo docker compose --profile agent up -d --force-recreate supervisor
```

At decision boundaries, SAM detects or tracks a noun phrase in the camera Muse
selected on the preceding decision. The first query defaults to the current
subtask target and first camera. Candidate boxes and clip-local tracks are
supplied to Muse; full segmentation output is retained in the audit trace. It does not decode
masks into a 3D pose or use SAM to modify GR00T images. Boxes alone cannot verify
contact, containment, or grasp success. SAM adds up to one API call per decision
and is skipped when only one call remains for the Muse decision.

## Run a closed-loop comparison

The agent loop pauses simulation while Muse responds. Each decision receives
current camera snapshots and the snapshots from the preceding supervisor call.
It chooses a bounded program, the dispatcher executes it in the same scene,
and the next decision receives fresh observations. This is synchronous closed-loop
supervision, not continuous frame-by-frame API streaming.

Start with a single atomic task and seed to check the entire pipeline:

```bash
sudo docker compose --profile agent build supervisor sim
sudo docker compose --profile agent run --rm --no-deps supervisor \
  python -m unittest discover -s tests -v
sudo docker compose --profile agent up -d inference supervisor
sudo docker compose --profile agent run --rm sim \
  python -m sim.compare \
    --task OpenCabinet --seeds 0 \
    --output /outputs/open-cabinet-comparison-001
```

For each scene seed, the comparison runs these conditions sequentially:

1. `baseline`: GR00T with the original task instruction, without Muse or SAM.
2. `muse`: Muse planning, visual verification, instruction updates, and replanning.
3. `muse_sam`: the same initial plan as `muse`, plus adaptive SAM image detection
   at decision boundaries. Add `--sam-video` to `sim.compare` to opt into clip
   tracking. Later decisions and replans can differ.

The runner checks both services' metadata before starting the baseline. This
does not call Meta or execute policy inference. Building an image and running
the disposable unit-test container do not start the persistent supervisor.
Use the `up -d --wait` command above before a comparison; it also starts a
missing service or recreates a service when its image has changed. If health
fails, inspect `sudo docker compose --profile agent logs --tail=80 inference supervisor`.

The task split, seed, episode horizon, and GR00T action chunk length match across
conditions. Direct gripper intervention is disabled in all conditions. Each agent
episode allows at most 90 decisions and 200 API calls by default. SAM consumes an
additional call per decision; actual use depends on early completion and stopping.
This can involve substantial API usage: start with one seed and inspect the trace.
Lowering `--max-api-calls` may end the SAM run earlier than the Muse-only run.

The runner writes `comparison.md` and `comparison.json` after every completed
condition. Each condition retains its video, plan, trace, and metrics under
`seed-N/CONDITION/`. `sam_calls_with_boxes` reports whether SAM found candidate
regions; it does not establish that those regions were correct or useful.
`subtasks_marked_complete` records Muse claims, not simulator task success.
Errors stop the comparison and are reported separately from completed failures.

Once the one-seed run works, use a new directory and more seeds:

```bash
sudo docker compose --profile agent run --rm sim \
  python -m sim.compare \
    --task MakeIceLemonade --seeds 0 1 2 \
    --output /outputs/lemonade-comparison-001
```

Review success alongside videos, stop reasons, steps, API calls, and wall time.
An atomic task probes recovery and verification; a composite task additionally
probes decomposition. The existing `fixed` mode can isolate the value of an
instruction sequence without feedback. Matched scene seeds do not make GR00T
sampling or Meta outputs identical, and a few episodes are diagnostic evidence,
not a reliable performance estimate. A run with SAM boxes but no success gain is
also a valid result; boxes alone add no grasp or motion-planning skill.

## On-demand information gathering and object memory

The default `--planning-context summary` adds a bounded episode-local notebook
to every supervisor request: original goal, initial/unfinished plan, accepted
revisions, completed-subtask claims, recorded executions, changed object locations,
and recovery feedback. It summarizes explicit evidence and events, not private
model reasoning, and resets for each episode. Current camera views, measurements,
and scene memory are still supplied. Inspect `episode_context.json` and the exact
request context in `trace.jsonl`; no extra summarization API call is made.
Use `--planning-context recent` to omit the notebook for an ablation, retaining
the existing recent-event window, accepted handoffs, and object memory.

All agent tasks also receive `execution_feedback`, pairing actual policy bursts
with the next valid Muse assessment. Instruction counters survive subtask renaming;
unchanged phase/grasp reports prompt review even if the robot keeps moving. These
are advisory model-claim comparisons, not automatic failure detection or motor
interventions. API failures/invalid proposals do not count as visual reviews.
`execution_feedback.json` saves the latest snapshot in the episode directory.

Scene memory retains an earlier `identity_anchor`, up to six changed observations
per entity, and the preceding location claim. These are fallible historical
claims for resolving appearance/location conflicts, not a verified object tracker.
Additional identity-history content in each request is budgeted to 24,000
characters; `identity_history_records_omitted` reports omitted records. Current
entries are retained, and internal history remains available for later requests.

Supervision checks the visible outcome relevant to each goal: acquisition during
withdrawal, support and release for placement, actual arrival for navigation,
the same articulated component for fixture operations, or the state change in
another task's success criteria. Policy instructions use the shortest unambiguous
object name, at most one essential modifier; detailed instance attributes and
source/destination relations stay in Muse's context. Policy inference does not
receive Muse's notebook or SAM boxes. Navigation, access, pickup, placement and
verification stages use policy language requests with observed outcome checks,
without inventing position commands or assuming standalone skills are reliable.
A contradicted completed
goal requires a new repair goal; the historical completion claim remains in the
trace. Guidance and retained evidence do not guarantee correct visual judgments.

There is no mandatory scouting tour. Each decision supplies a structured
assessment of transfer phase, possible held-object identity, grasp uncertainty,
destination, missing information, and evidence. Object/fixture memory is owned
by the simulator and persists through subtask changes and replans, with earlier
observations kept as time-stamped claims. It resets between episodes. Muse can
update semantic locations such as `counter beside sink`, rather than fabricate
precise 3D positions. Empty detections and occlusion are not proof of absence.

If more information is needed, Muse can:

- Return `inspect` with an empty program to query perception again without
  moving. Three repeated identical inspection requests at the same simulator
  step stop with `perception_stall`; API and decision budgets always apply.
- Choose `perception.target`, `perception.camera`, and `perception.mode`
  (`detect` or `track`) for the next SAM query. This switches attention from
  acquisition to the destination when appropriate, rather than fixating on the
  original subtask target. Cameras can be queried across successive decisions.
- Use `navigate(fixture="sink", steps=30)` to request a bounded GR00T navigation
  burst. The target must have visual memory evidence OR be an exact fixture name
  in the original task, name a supported fixture, and have an explicit information need. No geometric
  planner is added. The current manipulation remains pending; successful
  navigation cannot satisfy its completion criteria.

Navigation allows six bursts per episode by default (`--max-navigation-bursts`).
It targets observed/remembered or task-named fixtures rather than arbitrary
coordinates or unseen objects. An unseen task-named fixture remains unverified;
navigation trace events record `original_task_description` rather than
`visual_memory` as their source. During destination search, common named counter/
table roles from the task cannot be reduced to generic `counter`/`table` targets.
Muse must assess motion, target reacquisition, and retention of
any possibly held object after each burst. Destination search should not cause
a second source pick or substitute a distractor for the original object.

With `--sam --sam-video`, the runner retains up to eight RGB samples per camera, sampled
every eight control steps plus the latest endpoint. SAM track mode encodes the
selected camera samples as H.264 MP4 inside the supervisor container and submits
the clip to SAM 3.1; Muse receives that clip and frame-level tracking records.
`clip_steps` maps clip frames to simulator steps. This is sparse temporal evidence,
not every-frame streaming. A clip with only one sample uses image detection.
Tracking uses a concept prompt and produces clip-local object IDs; IDs are not
treated as globally consistent across clips or cameras. Raw masks remain in the
audit. Current-frame boxes are separated from earlier tracking records. Without
SAM, Muse still has RGB snapshots and persistent memory, but no SAM tracking.
SAM defaults to image-only detection, even if Muse asks for track mode. Video is
an explicit run setting so image and clip failures can be isolated. SAM cannot
change the simulator's cameras; requests contain copies of rendered observations.

SAM uses at most one call per decision and is skipped when only one call remains
for Muse. Track requests cost differently from images; the provider documents a
minimum 50-frame charge for video, including short clips. See
[SAM video requests](https://dev.meta.ai/docs/sam/segmenting). No automatic retries
or silent switch to a different evaluation condition is performed.

Each policy burst records action magnitudes, maximum changes in public robot
measurements, and mean image change. Four consecutive numerically stationary
bursts (`--stall-bursts`, zero disables) prevent repeating the same instruction.
The supervisor must inspect, select a distinct supported recovery, or stop;
two consecutive rejected motion requests end with `recovery_unavailable`.
Motion does not itself establish grasp or semantic task progress.

The supervisor's Muse output limit defaults to 8192 tokens, configurable with
`META_MAX_OUTPUT_TOKENS` in `.env` (1024–16384). Incomplete responses report the
provider's reason when available. They remain errors; increasing the cap does
not guarantee completion. An error artifact retains the last step and memory.
SAM requests default to a 180-second timeout (`META_SAM_TIMEOUT`, maximum 240);
the supervisor client allows 480 seconds for the combined SAM/Muse request.
Transient decision failures permit bounded execution of the current subtask's
last policy instruction, followed by a fresh Muse request. See
`--max-supervisor-failures` in [the simulation guide](../sim/README.md).

## GR00T control with visual planning and correction

Planning uses dependency-based order for independent transfers. A visible or
already-held object required by the full goal can be handled before an unlocalized
object from the provisional first step. The planning policy is object-agnostic:
it derives outstanding requirements and dependencies from each task and the
observed state. A useful out-of-order grasp is not a reason to discard the object.
True physical, safety, and explicit strict-order constraints still apply; two
transfers are not assumed interchangeable without checking their side effects.

Muse makes a reorder explicit with `status=replan`, an empty program, and a
complete replacement of the unfinished suffix using new IDs. Completed work
stays intact, every outstanding transfer stays represented, and no scene reset
or grasp release occurs during replanning. The prompt instructs Muse to finish
transport of a useful held object instead of returning to its old source or
claiming the skipped target complete. `replans_left` reports the remaining
replanning budget. This is prompt guidance plus the existing suffix-replan
mechanism; it does not establish better visual grounding or successful execution.
Stopping guidance compares the latest views with a five-burst kinematic history,
including renewed motion after a pause. Numerical motion is not labeled task
progress. Muse is told to reconsider stale stagnation claims when useful progress
resumes, and that zero navigation/replan budgets do not disable the current
GR00T transfer. Remaining execution options are explicit in its context.
`status=stop` remains a model decision honored by the runner, not a guarantee
that the task could not have succeeded with more execution.
The selected task below is a smoke test, not a task-specific controller branch
or evidence of generalization. Evaluate the same frozen prompts across held-out
tasks and seeds, including order-sensitive tasks; preserve per-task regressions
alongside improvements. The supervisor does not add manipulation skills to GR00T.

Use `--conditions baseline muse` to test without SAM first, then compare against
`muse_sam`. All conditions pass GR00T action values through unchanged, including
its predicted release actions. There are no grasp latches, base/orientation
overrides, or translation clamps in the CLI execution path. Merely seeing a
destination no longer imposes an eight-step feedback interval. Muse can choose
a shorter interval for a concrete ambiguity, but should preserve useful skills.

When GR00T acts on the wrong fixture, Muse compares current observations against
time-stamped scene memory and optional SAM candidates, requests supported fixture
navigation through GR00T, then verifies arrival before resuming the pending task.
SAM boxes and historical sightings do not provide metric routes or guarantee
that GR00T will follow a navigation instruction successfully.

Invalid model decisions return their audit and API usage without executing
motion or updating memory. The runner records `invalid_decision` feedback for
the next budgeted Muse request and stops after three consecutive invalid proposals.

The first selected task is `ArrangeTea`: kettle and mug onto a tray, followed by
closing cabinet doors. Its cabinet starts open. At the pinned RoboCasa revision
it belongs to `composite_unseen` and is excluded from `pretrain300`.
`--require-unseen-composite` checks both memberships inside the simulator before
execution. This is benchmark task holdout for the multitask checkpoint, not a
claim about every source of GR00T foundation pretraining. `--split pretrain`
keeps the benchmark's pretraining scene distribution; task holdout and scene
holdout are different dimensions.

Run on Thor, using a fresh output directory:

```bash
sudo docker compose --profile agent build sim supervisor
sudo docker compose --profile agent run --rm --no-deps supervisor \
  python -m unittest discover -s tests -v
sudo docker compose --profile agent up -d --wait inference supervisor
sudo docker compose --profile agent run --rm sim \
  python -m sim.compare \
    --task ArrangeTea --split pretrain --seeds 0 \
    --conditions baseline muse --require-unseen-composite \
    --max-decisions 190 --max-api-calls 200 \
    --output /outputs/tea-planner-comparison-001
```

Proceed only after tests and service health pass. The supervisor must use its
rebuilt image because the program vocabulary and placement instructions changed.
Inspect instructions, visual claims, and environment task success in `trace.jsonl`
and `result.json`. GR00T action values are unmodified in every condition.

## Scene and RGB integrity

The comparison runner fixes `PYTHONHASHSEED` for child processes. The simulator
also re-executes standalone `sim.run_agent` processes with that seed if needed,
before importing RoboCasa. Setting the variable after interpreter startup alone
would not fix Python's hash order. This also applies to reference-scene retries.
The simulator
seeds Python and NumPy and selects an explicit valid layout/style pair from the
requested split, applying the task's exclusions. It uses that singleton pair and
preserves the requested object-instance split when constructing RoboCasa.
All conditions then use the same spawn replay reset path: baseline records its
sampled mobile-base position/orientation, and supervised runs supply that pose
to RoboCasa's normal reset. Controllers and physics settling still initialize
normally; simulator joint state is not copied over a mismatching reset.

Every episode saves `scene.json`: layout/style, instruction, hashes of initial
episode metadata, model XML, and rounded simulator state, plus initial RGB hashes.
It also saves the initial metadata and numeric state for mismatch diagnosis.
Fingerprint version 2 sorts only each object's `info.groups_containing_sampled_obj`
membership list, whose upstream order depends on set iteration. Other lists,
asset paths, placement metadata, XML, and physical state remain strict. Version 1
references are accepted only when their saved metadata matches under that same
narrow normalization and every other identity field matches; the trace records
`scene_verified` and whether membership-order normalization was required. Raw
metadata remains saved for inspection.
Supervised conditions must match the baseline fingerprint before any actions or
Meta requests. A mismatch aborts the comparison rather than silently comparing
different scenes. The fingerprint is privileged evaluation metadata and is never
sent to the supervisor or policy. It does not seed stochastic GR00T/API sampling
or prove identical GPU-rendered pixels. Old results without scene fingerprints
cannot establish matched conditions. Use new output directories after this change.
References created before spawn metadata was saved must be regenerated. A
remaining mismatch reports changed metadata fields, maximum state delta, and
the first changed state indices; inspect both conditions' `scene.json` values.

Raw RGB arrays are checksummed in the simulator and verified after transport in
the supervisor. API image inputs use lossless PNG. `received_image_diagnostics`
in each supervisor response can be compared with the request's diagnostics.
If every current camera has extreme high-frequency pixel noise, a heuristic
guard stops before the next policy/API request and saves
`suspected-noise-step-*.npz`, per-camera PNGs, and `image_quality_error.json`.
Inspect these source pixels; the heuristic does not diagnose an EGL/driver bug.
If source images are clean but Muse reports noise, inspect API/media handling
rather than treating the text report as proof of simulator corruption.
`inspect` explicitly refreshes the declared sensors at the same robot pose without
advancing physics. It can refresh a stale render, but cannot provide a new vantage
point or repair an underlying driver/rendering failure.

Fixture memory keeps the last observed location and evidence even when current
visibility becomes uncertain, so the robot can revisit a previously seen sink.
Observed counter/table fixtures can be navigation language targets; this does not
establish policy reliability for them or add arbitrary-coordinate navigation.
While a grasp is supported and the destination is visible (or placement phase
is selected), motion bursts are capped at eight control steps by default
(`--placement-burst-steps`). Truncation is logged as `bounded_execution` and no
unused action queue survives. Muse is prompted to request placement rather than
another pickup from the old source. This increases feedback frequency; it cannot
guarantee retention, collision avoidance, or a successful placement.

Rebuild both containers because the shared decision schema now requires
`assessment`, `memory_updates`, and `perception` in addition to the existing
decision fields. The supervisor image installs FFmpeg inside the container;
no host installation is required. Run the fake-provider tests before a live trial:

```bash
sudo docker compose --profile agent build supervisor sim
sudo docker compose --profile agent run --rm --no-deps supervisor \
  python -m unittest discover -s tests -v
sudo docker compose --profile agent up -d inference
sudo docker compose --profile agent up -d --force-recreate supervisor
sudo docker compose --profile agent run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode agent --seed 0 --sam \
    --decision-steps 40 --max-decisions 120 --max-api-calls 200 \
    --max-navigation-bursts 6 --stall-bursts 4 \
    --output /outputs/lemonade-memory-001
```

Inspect `episode-000/scene_memory.json`, `trace.jsonl`, `perception-*.npz` camera
samples, and `rollout.mp4`. The trace exposes perception queries, SAM tracks,
navigation tools, memory updates, and rejected recoveries. Real task success
still comes only from RoboCasa. Navigation and tracking remain fallible and their
contribution must be measured against the same policy without these interventions.

## Artifacts and limits

Each episode directory contains:

- `plan.json`: initial short-horizon plan.
- `rollout.mp4`: uninterrupted first-camera video of the episode.
- `observation-*.jpg`: labeled visual evidence sent to the supervisor.
- `program-*.py`: generated command programs for review, interpreted by the
  bounded dispatcher rather than a general Python interpreter.
- `trace.jsonl`: plans, API response IDs and usage, raw model outputs, decisions,
  tool arguments, and executed step counts. Replans appear in this trace.
- `result.json`: environment-defined success, stop reason, steps, and timings.
- `error.json`: diagnostic details if an episode aborts.

Top-level `config.json` and `summary.json` record configuration and results.
Only the simulator's original task check can report full-task success. Finishing
the plan or a positive Muse verdict cannot mark a failed episode successful.
In agent mode, a final `complete` proposal while simulator success is false emits
`final_completion_blocked` and keeps the current stage active. Muse receives the
overall `simulator_success` flag and dated `final_completion_feedback` on subsequent
requests, including with recent-only planning context. It can continue policy
execution or replan; no automatic withdrawal or extra settling actions are injected.
Simulator success ends execution immediately. Existing step, decision, API, stop,
and failure limits still bound unsuccessful runs.

Public task requirements are supplied in `simulator_context.task.success_requirements`.
Currently GarnishPancake has verified rules from the pinned RoboCasa source,
including required contacts, centering, and gripper distance greater than 25 cm.
Other tasks explicitly report uncatalogued requirements and use the same authoritative
success gate. Individual predicate results and hidden object poses are not sent to Muse.
Privileged object state and live individual predicate results are not passed to Muse.
Success feedback and requirements are supervisor context, not additional policy inputs.

API calls, decisions, replans, per-subtask steps, per-program steps, and total
episode steps are bounded independently in `--subtask-budget-mode plan` (the
strict-budget option). In `--subtask-budget-mode episode` (the default), planned
`max_steps` values are advisory: the active goal can use the remaining episode
budget. Decision, API-call, replan, per-program, and episode limits still apply,
and Muse may stop early or complete a goal on visual evidence. The comparison
runner defaults to `episode` mode for both supervised conditions so a small
planned subtask ceiling does not terminate one condition prematurely. Override
with `sim.compare --subtask-budget-mode plan` to evaluate strict ceilings.
Muse output tokens are capped per call; SAM uses its segmentation response format.
The call budget is not a dollar cap; prices and input-token usage vary. API/model
errors abort rather than silently spending on retries. A failed request may have
been billed even when no usage record was returned. Keep exploratory budgets
small until the account and task behavior are validated.

## Investigate early stops and SAM errors

In plan budget mode, a failed 600-step pick after a 330-step door opening can end
the episode at 930 steps when Muse chooses `stop`. This is a subtask ceiling,
not the simulator's 3000-step horizon. `episode` mode permits more attempts but
cannot make a frozen policy execute a skill it has not learned. Inspect repetitive
commands and videos before interpreting longer execution as better recovery.

A completed SAM response with empty text means no matching region was found;
it is recorded with empty boxes and still passed to Muse. Incomplete responses,
refusals, HTTP failures, and timeouts remain errors. Upstream HTTP errors identify
the model and include a bounded, credential-redacted diagnostic. There are no
automatic retries or silent fallbacks to a Muse-only condition.

Before recreating a failed supervisor, preserve its diagnostic:

```bash
sudo docker compose --profile agent logs --tail=100 supervisor
```

To retry SAM + Muse alone using a previous initial plan, reset the same seed into
a new output directory:

```bash
sudo docker compose --profile agent run --rm sim \
  python -m sim.run_agent \
    --task MakeIceLemonade --mode agent --seed 0 --sam \
    --subtask-budget-mode episode --max-decisions 90 --max-api-calls 200 \
    --plan /outputs/lemonade-comparison-001/seed-0/muse/episode-000/plan.json \
    --output /outputs/lemonade-sam-retry-001
```

This begins a new episode; it does not resume the old robot state. A new full
comparison is required for matched results under the changed budget policy.

## Access and subtask transitions

The supervisor must provide structured `handoff.readiness` and supporting evidence
with every decision. Before advancing a completed subtask to the next one, the
runner requires `ready`. `blocked`, `uncertain`, and `not_applicable` cannot advance
an intermediate subtask; they generate a `transition_blocked` event without motion
and consume a supervisor decision. The supervisor can then continue access work
or replan. This guard does not use privileged simulator state and cannot prevent
an incorrect visual claim of readiness. Normal decision/API limits bound repeated
rejected completions. The final subtask has no next-step handoff requirement.

For a fridge-to-object sequence, the access criterion concerns the required
shelf/object, rather than merely one open panel. It does not require both doors
when one is sufficient. Release, reposition, and inspection can remain part of
access work. If access is lost later, the supervisor may insert a new access
prerequisite before the pending manipulation rather than treating all motion as
a grasp attempt. Prior completion and handoff evidence remains available after
the recent-history window rolls off, explicitly labeled as earlier model claims.

`subtask_transition` trace events record accepted transitions; `decision` events
record model claims, including claims rejected by the handoff guard. Rebuild both
`supervisor` and `sim` after changing this shared decision contract.

This is an initial experimental harness. It does not establish improved success
rates, reliable subtask verification, or generalization; those require matched
rollouts and review of failure traces.

References: [Meta Responses API](https://dev.meta.ai/docs/protocols/responses),
[structured output](https://dev.meta.ai/docs/structured-output),
[image understanding](https://dev.meta.ai/docs/image-understanding),
[SAM prompts and output](https://dev.meta.ai/docs/sam/segmenting).
