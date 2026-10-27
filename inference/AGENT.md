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
sudo docker compose --profile agent up -d inference supervisor
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

Add `--allow-gripper` only when testing direct gripper interventions. This makes
the controller-checked tool available; it does not guarantee a valid grasp or
prevent a held object from falling when released. No world-space corrections are
issued by this primitive.

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

Set `META_USE_SAM=1` in `.env`, then recreate only the supervisor:

```bash
sudo docker compose --profile agent up -d --force-recreate supervisor
```

At decision boundaries, SAM segments the current subtask's target noun phrase
in the first policy camera. Its candidate boxes are supplied to Muse; full
segmentation output is retained in the audit trace. This version does not decode
masks into a 3D pose or use SAM to modify GR00T images. Boxes alone cannot verify
contact, containment, or grasp success. SAM adds up to one API call per decision
and is skipped when only one call remains for the Muse decision.

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
Privileged object state and task predicates are not passed to Muse or GR00T.

API calls, decisions, replans, per-subtask steps, per-program steps, and total
episode steps are bounded independently. API output tokens are capped per call.
The call budget is not a dollar cap; prices and input-token usage vary. API/model
errors abort rather than silently spending on retries. A failed request may have
been billed even when no usage record was returned. Keep exploratory budgets
small until the account and task behavior are validated.

This is an initial experimental harness. It does not establish improved success
rates, reliable subtask verification, or generalization; those require matched
rollouts and review of failure traces.

References: [Meta Responses API](https://dev.meta.ai/docs/protocols/responses),
[structured output](https://dev.meta.ai/docs/structured-output),
[image understanding](https://dev.meta.ai/docs/image-understanding),
[SAM prompts and output](https://dev.meta.ai/docs/sam/segmenting).
