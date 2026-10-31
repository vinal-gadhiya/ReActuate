# Shared RoboCasa agent harness

This harness targets all RoboCasa task families. Task names and objects belong in
public reference catalogs and evaluation fixtures, not special-case controller
branches. Policy observation formats and numerical actions remain unchanged.

## Episode flow

1. `sim/run_agent.py` resets the original environment and records the scene,
   instruction, policy metadata, RGB observations and robot measurements.
2. `inference/supervisor.py` asks Muse for a scene survey, the full set of original
   requirements, and a short action window. These share one planning request.
3. `core/goals.py` stores immutable requirement IDs, descriptions and dependencies.
   Statuses start unknown. Every stage references `goal_ids`; replanning can
   replace stages but cannot remove or redefine requirements.
4. Muse assesses each active stage as satisfied, unsatisfied or unknown using
   actual camera/step references. Optional transport details describe grasp and
   carrying when relevant. Washing, heating, switching, pouring and stirring do
   not have to masquerade as transfer phases.
5. One validated literal tool call executes for a bounded number of steps.
   `run_groot` invokes the selected frozen backend, either GR00T or PatchWAM.
   No generated Python is evaluated. The CLI exposes no manual motor overrides.
6. Each actual execution receives an attempt ID. Muse reviews its observed
   outcome; attempt history belongs to goal IDs, not replaceable stage IDs.
7. Completing a short window with pending requirements requests another window.
   Window extensions do not consume recovery replans, but decision, API and
   episode budgets still apply. Intermediate transitions require ready handoff.
8. The simulator defines the reported task score. Visual completion claims do
   not set `success=True`. The default CLI `--execution-policy episode` continues
   a valid committed stage until simulator success/end or the episode step limit.
   A genuine execution error still fails the run. `--execution-policy bounded`
   retains early model-stop and short retry-limit behavior for comparisons.

## Stage commitment and episode budget

One 32-step burst is a feedback interval, not an entire skill budget. Episode mode
keeps the active stage's instruction through unsuccessful/uncertain observations.
Its planned `max_steps`, retry/recovery counters, and stationary diagnostics do
not terminate physical execution. API and decision limits still cap model calls;
after they are exhausted, the runner continues the committed instruction without
claiming completion or advancing to an unverified next stage. Continuations are
logged as `policy_continuation` and ordinary tool/attempt events. A temporary
recovery command does not replace the committed stage during fallback.

Changing instructions before stage completion requires current evidence and a
specific `recovery.necessity`: blocked access, wrong target, object loss, unsafe
state, goal contradiction, or a useful required object already held. Failed
bursts alone and switching tool aliases do not establish necessity. The judgment
is still a model claim; the harness validates its structure/references, not visual
truth. Replanning executed work requires the same intervention evidence. Stage
completion requires a satisfied outcome with current observation evidence.

When a completed action window has pending requirements, another validated window
is required. If supervision is unavailable at that boundary, the harness reports
the supervision limit; it does not invent a next instruction or replay a completed
window. Maintain enough supervision budget for the intended number of stages.
Model stop and malformed proposals during an unfinished stage do not end it.

For a 2,000-step physical budget, use `--max-steps 2000 --execution-policy episode`.
Rebuild and recreate both supervisor and simulator images for the new recovery
schema and execution policy. This change grants execution time and controls
switching; it does not guarantee that a frozen policy will produce useful motion.

The ledger preserves the initial extraction. It cannot establish that Muse
extracted every requirement correctly, identified objects correctly, or judged
completion accurately. Those are model-quality questions requiring evaluation.

## Instruction boundary

`core/instructions.py` separates semantic action fields (intent, object,
destination, instruction) from backend language rendering. Explicit instructions
reach either backend verbatim. The executable command must match the rendered
text. For the version-2 agent, instructions must match the atomic catalog's
documented descriptions or its `instruction_forms`, with scene entities substituted.
Forms are derived reference data: split pickup/place clauses, navigation, fixture
opening/closing and switching. Specialized descriptions retain their documented
button, lever, rack and temperature wording. Documented alternatives and singular/
plural door notation may be instantiated. Case and final punctuation may differ;
accepted instruction text is never rewritten. No 12-word/96-character limit or
panel-name ban is imposed.

The atomic catalog has descriptions and categories. `skills_required` and
`skill_keys` come from the composite catalog. Harness-authored `skill_reference_tasks`
associations in the atomic catalog reserve relevant examples for those skill
families before lexical retrieval. These associations and the instruction forms
are labelled derived data, not additional upstream annotations or trained skills.
Muse receives the current composite reference, relevant atomic descriptions and
the complete shared form list on each request.

Generated initial plans, replacement windows and executable instructions are
checked before motion. Invalid language produces budgeted correction feedback;
it is not silently paraphrased. For example, `Grab the mug.` must be corrected to
`Pick up the mug.` A tool trace records the matched form, source atomic task(s),
and catalog revision. Baseline and fixed-mode instructions remain unchanged.

```python
run_groot(instruction="Pick up the mug.", steps=32)
run_groot(instruction="Navigate to the dining counter.", steps=32)
run_groot(instruction="Place the mug on the tray.", steps=32)
```

These are three separate decisions. Compatibility `navigate(fixture="dining
counter", steps=32)` renders `Navigate to the dining counter.` Named destinations
may be requested without prior sighting or original-task mention. Arrival remains
a visual claim. In episode mode the physical horizon applies; short navigation
and stationarity limits remain available with bounded execution policy.

Backend profiles currently use the same simple fallback phrasing. They provide a
boundary for measured changes, not evidence that both policies perform equally.
Legacy library clients retain compatibility validators; the version-2 CLI adds
catalog-language validation to semantic command matching. Regenerate old plans
that contain free-form instructions after rebuilding both `sim` and `supervisor`.
Matching a language form does not prove correct object grounding or that the
policy can execute the subtask; those still require paired rollouts.

## Persistent outcomes and recovery

Requirement updates need evidence and actual observation references. Satisfaction
requires current evidence; reopening a satisfied requirement needs newer
contradictory evidence. Declared dependencies are checked. Unknown goals,
cyclic dependencies, duplicate identities and stages referencing absent goals
are rejected.

Attempt outcomes are `intended_object_acquired`, `other_required_object_acquired`,
`unrelated_object_acquired`, `progress`, `prerequisite_changed`,
`no_reported_progress`, or `uncertain`. These are fallible reports, never private
simulator labels. Generic progress applies across task families. Uncertainty is
not counted as confirmed failure.

With `--execution-policy bounded`, after `--max-goal-failures` supported unsuccessful outcomes (default 3), execution
requires a recovery proposal. A strategy must change the actual normalized
command; a prerequisite retry must cite a subsequent executed action reported as
changing that prerequisite. Previously used prerequisite evidence cannot justify
repeated retries. `--max-goal-recoveries` defaults to 2. Stage renaming and burst
size changes do not reset history. Language normalization is only an accounting
key; it never changes executed text. The harness cannot prove that a different
sentence is a genuinely different physical strategy.

Episode mode retains this history for diagnostics, but failure/recovery allowances
do not terminate or force switching of the committed stage. Before/after references
are accepted together when an after-action reference is present. Repeating the
same supported outcome with new wording updates its evidence history without
adding failures; changing a supported outcome to reset its count is rejected.

## Working context and perception

`core/working_context.py` sends the original goal, full ledger, active/next stage,
current and previous observations, relevant dated beliefs, recent attempts,
recovery feedback, robot measurements and budgets. The full trace remains on
disk; overlapping episode summaries, surveys and histories are omitted from the
model working view. Beliefs include age and evidence, not ground-truth poses.

The public composite catalog covers 300 tasks, with descriptions and Skills
Involved labels. The current task's entry accompanies the episode. Up to ten
relevant atomic descriptions are retrieved from the 65-entry public catalog.
Omitted references do not disable actions. Catalog skill labels describe tasks,
not verified policy capabilities or compulsory action order.

`--depth` supplies aligned, fixed-scale surface-depth visualizations to Muse only.
The policy still receives its original RGB/proprioception contract. Depth does
not directly measure contact, grasp or gripper-object distance. Raw depth and
visualization metadata are retained. SAM uses RGB; candidates and tracks remain
perception hypotheses. Neither depth nor SAM supplies privileged success labels.

Saved version-2 plans include requirement definitions. A new episode resets
statuses, never resumes old physical state, and treats saved surveys as historical.
Legacy saved agent plans trigger fresh requirement extraction with the supplied
plan as a candidate. Fixed mode continues accepting legacy scheduled plans.

## Artifacts and evaluation

Episode artifacts include `trace.jsonl`, observations, rollout video, `plan.json`,
`scene.json`, `simulator_context.json`, `goal_ledger.json`, `attempt_history.json`,
and legacy audit summaries. Full requirement-update history is also in the trace.
`tool.policy_instruction` records the actual backend instruction. Budgets and
stop reasons distinguish controller stops from simulator-defined success.

`sim.compare --tasks TASK_A TASK_B ...` runs a suite with common settings and
matched initial scenes. Child failures are recorded and the suite continues;
conditions lacking a matching baseline scene are skipped explicitly. Reports
include completed-rollout success, success among all attempted episodes,
operational error categories, and verified paired outcomes: both succeed, both
fail, agent improvement, baseline-success regression. Skipped dependent runs are
not attempted episodes. Startup service preflight failures still fail the command.

`sim.review_cases` extracts real request snapshots into unlabelled manifests:

```bash
python -m sim.review_cases --extract /outputs/RUN/episode-000/trace.jsonl \
  --output /outputs/cases-unlabelled.json
python -m sim.review_cases --cases /outputs/cases-annotated.json \
  --output /outputs/case-results.json
```

Assign whole episodes to development or held_out partitions. Annotate multiple
acceptable `eligible_first_targets` for planning, or offline outcome labels for
completion judgments. Labels are never sent to Muse. Unlabelled cases make no
API calls. Replay reports eligible-choice rate, false positives and abstentions;
it makes real API calls but executes no robot actions. Saved JPEG snapshots can
be lossy. Offline labels must match the active stage outcome being measured.
Replay is not closed-loop success evidence.

Validation gates: contract tests, labelled real-observation cases, then paired
rollouts spanning task families, including baseline-success regressions and
operational failures. Mocked tests establish plumbing, not model competence.
No new live performance results are claimed by this refactor.

## Thor deployment and checks

Run Python, tests, builds, simulation and inference on Thor containers only.
For the PatchWAM compose configuration, rebuild both shared-source services:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent build sim supervisor
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent run --rm --no-deps supervisor \
  python -m unittest discover -s tests -v
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent up -d --no-deps --force-recreate --wait supervisor
```

For GR00T omit the PatchWAM override. These source changes do not require
rebuilding policy weights. Use fresh output directories and keep policy versions,
seeds, depth/SAM settings, budgets and scene identities in comparison artifacts.
