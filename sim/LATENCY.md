# Supervision latency experiment

Rollback reference: `dcfb9e66f6a6901dd80036feb17cc904e1faff81` on main before
these edits. Changes are task independent; no frozen policy inputs, action values,
instruction text rewriting, control intervals or episode horizons change.

The previous schedule generally called a visual reviewer before the planner.
Selective verification runs the planner first. Ordinary continuation uses one
Muse call; a consequential claim adds one independent verification call. SAM,
when explicitly enabled, and bounded output-limit retries can add calls.
This reduces redundant serial requests, not the physical step budget.

Completion, necessary interventions, supported grasp acquisition/loss/identity
changes, loss memory claims, changed final requirement status and consequential
attempt reports trigger verification. An unconfirmed claim becomes inspection;
its program, recovery, replacement plan, memory updates and ledger updates are
withheld. Dated feedback reaches the next planner request. Verification still
uses every supplied current/before/historical camera and optional depth. The
planner also retains that full evidence; frames are not dropped to gain speed.

Compact planning context now includes bounded dated identity trails, the last
transport claim and a short execution summary. General guidance covers observed
source grounding, cross-camera occlusion, useful progress, prerequisites and
cause-based workarounds across manipulation, navigation and appliance skills.
Illustrations are reasoning examples, not compulsory execution recipes. They
introduce no model calls, task-name branches or forced motor corrections.

RGB encoding is reused within a request. Each successful API audit entry now
records its role and wall seconds. Traces include verification reasons, feedback
and image counts; comparison episode metrics include planner/verification calls,
failed supervisor requests and supervisor wall time including failed requests.
Videos, exact observation files and complete audit logs remain available.

## Run on Thor

Run these commands in the matching checkout on Thor. Local tests, builds and
simulation are prohibited by the repository instructions.

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent build sim supervisor
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent run --rm --no-deps supervisor python -m unittest discover -s tests -v
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent up -d --no-deps --force-recreate --wait supervisor inference
```

Start with a small cross-family paired sweep, retaining baseline failures and
successes, operational errors, partial outcomes and false recovery/completion
claims. Identical task/seed pairs use matched baseline scenes in `sim.compare`.
The comparison runner now accepts the same unlimited supervision settings as
`run_agent`; 4,000 physical steps remain the episode horizon.

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent run --rm sim \
  python -m sim.compare \
  --tasks BreadSelection BlendIngredients HeatMultipleWater \
  --conditions baseline muse --seeds 0 1 \
  --expected-policy patchwam --action-steps 16 --decision-steps 32 \
  --execution-policy episode --max-steps 4000 \
  --max-decisions 0 --max-api-calls 0 --max-replans -1 \
  --verification-mode selective --output /outputs/latency-selective-001
```

Repeat with `--verification-mode always` and a different output directory for a
schedule ablation. Both modes use the new compact context and guidance; to compare
the full earlier implementation, also run the pinned rollback commit in a separate
checkout/image. Do not rebuild a shared supervisor while another experiment runs.
The original overnight queue automatically uses selective mode after rebuilding;
override it in queue `common_args` with `--verification-mode always` if needed.

Compare task success and baseline-success regressions on the same seeds, all-run
operational failures, physical steps, total wall time, supervisor wall time and
API counts. Inspect false loss and completion claims in all camera views. Lower
API counts are a plumbing expectation, not evidence of task performance. Accept
the change only after paired runs show lower wall time without observed completion
regressions; expand seeds if results are mixed. This source change alone cannot
guarantee equal or improved model quality.
