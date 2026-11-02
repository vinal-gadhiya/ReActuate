# 2953296 full-access and transfer experiment

Branch: `experiment/2953296-full-access-transfer`.
Base: `experiment/2953296-evaluation` at
`29532967b1dd96ec9b6b85fe1f34db3c43cac431`.

This experiment changes two shared behaviors across tasks:

- Opening stages must open all doors of the named fixture. For a paired-door
  fridge or cabinet BOTH doors must be open, each with current evidence and
  usable access. The confirmation response inventories the configuration and
  individual panel states. A claimed paired-door completion with a missing,
  closed, partly open, unknown or stale partner is rejected before stage advance.
  A single-door fixture needs its one door; a named drawer needs that drawer
  fully open. Unrelated fixtures, drawers and freezer compartments are excluded.
- New pick-and-place work first uses one complete source-to-destination atomic
  instruction, such as `Pick the mug from the counter and place it on the tray.`
  Navigation stays inside that policy request. Final placement/release completes
  the stage; acquisition or arrival alone does not. Only observed wrong behavior
  or a necessary evidence-backed recovery permits decomposition into pick,
  navigate and place. Retain original requirement IDs and destination, and skip
  phases already achieved. The new generic `execution_failure` intervention reason
  allows a supported strategy change without inventing an obstruction or loss.

Source-to-destination forms with `on`, `in` and `under` are derived from the
existing cited atomic descriptions, with source/destination entity substitutions.
The existing catalog validator applies to initial plans, replans and executable
requests. Unrelated appended actions remain rejected. No task-name branches,
frozen policy weights, numerical actions, image inputs or instruction rewriting
are introduced. Visual configuration and panel states remain model claims;
schema checks cannot establish the physical truth of a misidentified fixture.

## On Thor

Commit and push this branch from its worktree, then fetch/switch on Thor:

```bash
git fetch origin
git switch experiment/2953296-full-access-transfer

sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent build sim supervisor
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent run --rm --no-deps supervisor python -m unittest discover -s tests -v
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent up -d --no-deps --force-recreate --wait supervisor inference

sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent run --rm sim \
  python -m sim.run_agent --task ArrangeBreadBasket --mode agent \
  --split pretrain --seed 0 --episodes 1 --expected-policy patchwam \
  --require-unseen-composite --execution-policy episode \
  --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --max-decisions 190 --max-api-calls 200 --max-replans 20 \
  --planning-context summary --boundary-review --no-sam \
  --output /outputs/2953296-full-access-transfer-ArrangeBreadBasket-001
```

Use `--sam` instead of `--no-sam` for the optional SAM condition. Keep boundary
review enabled to exercise panel confirmation. Compare with the base branch on
matched scenes across task families, including paired doors, single doors and
drawers. Report task success, regressions, recovery outcomes, operational errors
and wall time. This branch retains the older commit's execution/review schedule
and finite API budget; it does not import the later main-branch latency or
completion-dispute changes. Python tests and simulations must run in Thor
containers, not on the local source-work host.

Output-limit robustness fix: a Muse `incomplete/max_output_tokens` response now
retains its usage audit and receives at most one bounded retry per supervisor
request, with a larger output allowance (capped at 16,384) and concise-output
instructions. Retries respect the shared API budget and preserve the requested
schema, including opening-panel confirmation. If no complete response arrives,
the runner retries supervision without robot motion and reports a structured
failure when its configured failure allowance is exhausted. Partial JSON is
never treated as a decision. Rebuild both sim and supervisor for this fix.
