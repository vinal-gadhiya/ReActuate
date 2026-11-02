# 244b087 with simulator-owned termination

This experimental branch starts at `244b087`. It keeps that version's single
Muse planner, optional SAM, visual memory, instruction validation and frozen
policy integration. It does not add the current main branch's goal ledger,
separate boundary reviewer, completion confirmation or extra historical images.

`sim.run_agent` and `sim.compare` now default to `--execution-policy episode`:

- Simulator success ends the episode immediately; otherwise the physical
  `--max-steps` budget governs normal episode length. Actual environment terminal
  states and execution errors are still respected and reported.
- Muse `stop` responses no longer terminate the episode. The last executed
  instruction for the current stage is continued for one bounded burst, then
  Muse receives fresh observations. Before the first action, the current planned
  instruction is used. Explicit instruction text and policy actions are preserved.
- Total decision, API-call, recovery replan and navigation burst caps do not end
  or disable supervision in episode mode. The saved run config resolves decision
  and API budgets to 0 and replans to -1. The protocol's 200-call allowance refreshes
  per request; it is not a total episode budget. No API-budget exhaustion fallback
  is entered during a healthy CLI episode run.
- Repeated inspections, invalid proposals, rejected actions, blocked handoffs
  and transient supervisor failures produce logged policy continuation rather
  than a controller-driven early exit. Permanent API/authentication, renderer,
  CUDA and other execution failures remain errors, not fabricated task results.
- If Muse completes the final stage, its visual claim is logged and Muse is
  asked to reassess original requirements and replan remaining work. Bounded
  continuation uses the original episode instruction, rather than replaying a
  completed final stage. A completed model plan never sets task success.
- Three supervision cycles without any physical progress trigger a policy burst,
  preventing infinite no-motion inspection/replan loops.

Use `--execution-policy bounded` to select the original early-exit behavior for
an ablation. Existing bounded-mode tests remain unchanged. In bounded mode,
positive decision/API budgets are honored; the total API budget can now exceed
200 (for example 500), because each request receives at most 200 remaining calls.
Decision/API values of 0 disable those total caps; replans=-1 disables that cap.

On Thor, from this branch, rebuild and run the regressions:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent build sim supervisor
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm --no-deps supervisor python -m unittest discover -s tests -v
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  up -d --no-deps --force-recreate --wait supervisor inference
```

One agent episode:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm sim python -m sim.run_agent \
  --task BlendIngredients --mode agent --split pretrain --seed 0 --episodes 1 \
  --execution-policy episode --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --planning-context summary --no-sam \
  --output /outputs/244b087-persistent-BlendIngredients-001
```

The older branch does not have `--expected-policy` or `--boundary-review`.
Use its PatchWAM Compose overlay and verify `config.json` identifies PatchWAM.

For baseline, Muse and Muse+SAM on the same scene seeds:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm sim python -m sim.compare \
  --task BlendIngredients --conditions baseline muse muse_sam --seeds 0 \
  --execution-policy episode --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --output /outputs/244b087-persistent-BlendIngredients-comparison-001
```

Keep the original unmodified `244b087` results distinct from this experimental
branch. Controller tests verify plumbing, not real policy competence: validate
on paired scenes and report baseline regressions, actual success, wall time,
and infrastructure errors separately. Use task-specific official horizons when
making benchmark-comparable claims instead of the custom 4,000-step setting.

## General recovery reasoning and source grounding

Recovery guidance applies to every action type and task. The same single Muse
planner compares each stage's intended visible outcome with its dated execution,
before/after evidence and reported scene updates. Feedback includes this context
for doors, lids, buttons, knobs, levers, racks, navigation, pouring and transfers;
it does not equate unchanged gripper/phase labels with a failed fixture operation.

Muse must reason about a supported cause before proposing a workaround: access,
viewpoint, wrong entity/component, missing prerequisite, obstruction, occupied
workspace, destination localization or stale state. Candidate strategies include
obtaining a useful view, completing a prerequisite, clearing an obstruction,
reordering eligible work or temporarily rearranging the scene. These are planning
options, not compulsory recipes or task-name branches. Each change must address
observed evidence, preserve the original requirements and useful progress, and
have an observable effect before the original work resumes. A known prerequisite
can be corrected immediately; more disruptive workarounds require meaningful
repeated failure evidence. Two observation bursts alone are not two failed skills.

The controller records and forwards these claims; it does not automatically
diagnose failure or select a workaround from string comparisons. This preserves
the older single-planner architecture and avoids extra review-model calls.
`tests/test_general_recovery.py` covers feedback across these action families;
live paired-scene evaluations are still required to establish model capability.

### Object-selection example of the general process

For acquisition, the planner prefers short pickup requests with an observed source, using
pickup-only clauses derived from atomic transfer descriptions:

- `Pick the mug from the tray.`
- `Pick up the mug from the blue tray.`
- `Pick the kettle from the counter.`

Short observed object/source qualifiers are allowed. Unknown sources remain
unknown; the planner may inspect or use a simple unqualified pickup instead of
inventing a source or color. Duplicate articles are rejected with feedback,
never silently rewritten. Policy action arrays and baseline instruction text
remain unchanged. The existing 12-word/96-character stage limit remains.

`execution_feedback.json` now includes `pickup_observations`: the requested
object/source extracted from executed language, actual execution/review steps,
and Muse's subsequent held-object/grasp claims. These are not environment truth.
`held_claim_epoch` tracks continuity through repeated bursts and occlusion. A
same-pose empty-grasp reassessment, identity-name change, or uncertain view does
not create a new acquisition. A later empty-grasp claim followed by a supported
hold begins a new epoch. Compare the saved images before interpreting these
claims as two separate physical acquisitions.

Muse is instructed to consider explicit distractor staging only after at least
two separate executed pickups are visually supported as selecting the same wrong
object. For an initial error it should first correct source/identity grounding
and return the held object to a suitable support if needed. A later workaround
may place the already-held distractor at a suitable observed location, navigate
back if necessary, and retry the actual target. It must preserve unfinished
requirements and strict task constraints, avoid unnecessary travel, record the
new location, and include restoration where required and permitted. This is a
model-planned strategy, not an automatic object-name comparison or guaranteed
perceptual/physical capability. There is no new recovery schema, additional
review model call, or hand-coded object-specific motion routine.

Rebuild **sim and supervisor**, run the tests, and recreate the supervisor after
updating this branch. To test the changed grounding and recovery on one scene:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm sim python -m sim.run_agent \
  --task BreadSelection --mode agent --split pretrain --seed 0 --episodes 1 \
  --execution-policy episode --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --planning-context summary --no-sam \
  --output /outputs/244b087-grounded-BreadSelection-001
```

Review `execution_feedback.json`, `trace.jsonl`, the source-qualified
`policy_instruction`/tool arguments, `plan_replaced` events, and the camera
images. A proposed workaround is not proof of recovery: verify the correct
object is actually acquired and the environment task succeeds. Compare the same
scene seeds on other object-selection tasks to check that the change generalizes.
