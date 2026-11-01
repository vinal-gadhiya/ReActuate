# Overnight experiment queues

Run these commands on Thor from the repository root. Each experiment starts a
fresh simulator process; inference and supervisor services stay running. An
experiment finishing, crashing, or failing its task does
not prevent the next experiment from starting. Ctrl+C in an attached batch stops
the queue and terminates its active child; it does not skip to the next job.

Build the simulator image to include the runner and example queue, and start the
shared services (use the current rebuilt supervisor image for recent fixes):

```bash
docker compose -f compose.yaml -f compose.patchwam.yaml build sim
docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  up -d --wait inference supervisor
```

Launch the batch detached so it survives closing your terminal or SSH connection:

```bash
docker compose -f compose.yaml -f compose.patchwam.yaml \
  run -d --name overnight-001 --no-deps sim \
  python -m sim.batch \
    --queue /app/sim/overnight.example.json \
    --output /outputs/overnight-001
```

The example runs twelve agent experiments: two seeds each for ArrangeBuffetDessert,
BeverageSorting, LineUpCondiments, CoolBakedCake, HeatMultipleWater, and
ToasterOvenBroilFish. All six tasks run with seed 0 before repeating with seed 1,
so the queue covers every task before spending time on repeats. Each has a
4,000-physical-step budget and no wall-clock deadline. The example also disables
total supervisor decision/API-call caps and recovery replan caps. A job may take
as long as needed; successful jobs finish early, while unfinished jobs can use
the full physical-step budget. Actual execution errors or unrecoverable service
failures are recorded and the queue moves on. Completed jobs move on immediately.
This queue has no whole-batch deadline and may run beyond the morning.

Watch progress without stopping the batch:

```bash
docker logs -f overnight-001
```

Results appear on the host in `outputs/overnight-001/`:

- `batch-summary.md`: quick status, actual environment task success, elapsed time.
- `batch-summary.json`: commands, exit codes, errors, steps and stop reasons.
- `queue.json`: exact queue used for this batch.
- `<experiment>/run.log`: complete stdout/stderr for that experiment.
- `<experiment>/run/`: normal videos, observations, traces, config and result files.

`completed` means the process finished and produced an episode result. It does
not mean the task succeeded: check `task_success`. A crash is `failed`, and
future experiments start as `pending`. There is no timeout unless you explicitly
set `--timeout-seconds` or a job's `timeout_seconds`; an explicitly timed-out job
is recorded as `timeout`.
The JSON summary is replaced atomically as progress is saved. Existing batch
directories are never overwritten. Choose a new container name and output path
for another batch. This runner does not automatically resume interrupted queues;
copy the pending jobs into a new queue and use a new output directory.

Customize `sim/overnight.example.json` before building, or mount your queue file
read-only into the batch container and pass its container path with `--queue`.
`common_args` apply to every experiment. Each experiment needs a unique `name`
and a `task`; optional fields are `seed` (default 0), `mode` (default `agent`),
`args` (additional `sim.run_agent` flags), and an optional `timeout_seconds`
(overrides the batch default; absent or null means no deadline by default).
For example:

```json
{
  "common_args": ["--expected-policy", "patchwam", "--max-steps", "4000",
                  "--max-decisions", "0", "--max-api-calls", "0", "--max-replans", "-1"],
  "experiments": [
    {"name": "garnish-baseline", "task": "GarnishPancake", "mode": "baseline", "seed": 0},
    {"name": "garnish-agent", "task": "GarnishPancake", "seed": 0,
     "args": ["--decision-steps", "32", "--no-sam"]}
  ]
}
```

Each job runs one episode. Use separate entries for multiple seeds. Task, mode,
seed, episode count and output directory are managed by the runner and cannot
be overridden in `args`. The runner passes other explicit arguments unchanged.

Shared PatchWAM RNG is not reset between jobs. These are independent diagnostic
runs, not guaranteed matched-scene or matched-policy-RNG comparisons; use
`sim.compare` for its matched-scene workflow. Run the batch regressions inside a
Thor container with the current source mounted:

```bash
docker compose -f compose.yaml -f compose.patchwam.yaml run --rm --no-deps \
  -v "$PWD:/workspace:ro" -w /workspace sim \
  python -m unittest tests.test_batch -v
```
