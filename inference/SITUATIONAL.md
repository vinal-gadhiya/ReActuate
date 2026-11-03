# Awomo direct Muse controller experiment

Branch: `experiment/awomo-situational`, from
`experiment/2953296-full-access-transfer` at `6857f84`.

On this branch, `sim.run_agent --mode agent` with Awomo defaults to
`--agent-control situational`. Muse receives the original task, labelled current
and temporal images, the last instruction, recent decisions/executions and atomic
task references. It chooses one instruction and its bounded execution interval.
It can change instructions on every feedback interval without a recovery label,
subtask completion, or boundary approval. Initial stage planning, stage ledgers,
boundary detection, completion confirmation and separate release-readiness
verification do not run. There is one Muse controller call per interval in the
normal path, plus optional SAM and a bounded retry for truncated output.

Atomic action templates and vocabulary are still validated. Muse may choose
separate pick/navigation/place instructions or a documented complete transfer;
neither decomposition nor full-transfer-first is enforced. Instructions reach
Awomo verbatim. Named source/destination slots can include observed distinguishing
attributes. The original goal is never replaced by these mutable instructions.

`Open the gripper.` remains a short language instruction (at most one policy
chunk). Muse judges alignment and low release height itself from the images;
there is no independent authorization call in this experiment. No numeric motor
override is introduced. The policy sensors, history, weights, action mapping and
16-step cadence remain unchanged. The simulator decides task success/end and the
physical step limit. Muse has no stop/completion action. API/decision caps default
to unlimited; explicit caps and repeated infrastructure/invalid-response errors
still produce an explicit operational failure instead of replaying stale actions.

`--boundary-review` is disabled in this mode even if supplied. Use
`--instruction-mode atomic` or `auto`; relational mode is a separate experiment.
`--plan` and `--plan-only` are unavailable because there is no stage plan.
The inherited stage/recovery/inspection limits do not govern situational control.
The full parent controller is available explicitly with `--agent-control staged`,
including its relational language mode and independent verification.

## Thor commands

After committing/pushing this branch and fetching/checking it out on Thor:

```bash
sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent build sim supervisor

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent \
  run --rm --no-deps supervisor python -m unittest discover -s tests -v

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent \
  up -d --no-deps --force-recreate --wait --wait-timeout 1800 inference supervisor

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent run --rm sim \
  python -m sim.run_agent --task GatherTableware --mode agent \
  --split pretrain --seed 0 --episodes 1 --expected-policy awomo-0.5i \
  --agent-control situational --instruction-mode atomic \
  --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --max-decisions 0 --max-api-calls 0 --no-sam \
  --output /outputs/awomo-GatherTableware-situational-001
```

Replace `--no-sam` with `--sam` for optional visual grounding. To compare against
the parent controller, use `--agent-control staged --instruction-mode relational
--boundary-review` and a new output directory. Use the same scene/seed; supply
`--reference-scene <baseline-output>/episode-000/scene.json` when available.
Baseline mode continues to use the original task instruction.

Videos, labelled observations, API audit, `situational_decision` and tool events,
configuration, scene identity and results are saved in the usual output folder.
`plan.json` has an empty steps list in situational mode, not a simulated plan.
Inspect the resolved `agent_control` in config/result to identify the ablation.

Evaluate matched scenes across manipulation, navigation and appliance tasks.
Report success, incorrect extra actions, regressions, wall time, API calls and
infrastructure failures. Fewer normal-path API calls do not establish improved
task quality. The mock tests check contracts, not Awomo/Muse competence.
