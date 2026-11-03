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
- New pick-and-place work is decomposed from the start: pick from an observed
  source, navigate only when actual travel is required, and place at the original
  destination. Helper stages retain the final requirement IDs; acquisition alone
  does not satisfy final placement. Skip phases already achieved. Generated plans
  with combined pick-and-place stages are rejected in atomic instruction mode;
  explicitly supplied older full-transfer instructions remain supported.
  Awomo now defaults to relational instruction mode, which permits a concise
  combined transfer when observed references make it useful. See
  [Awomo's language experiment](../inference/AWOMO.md#relational-instruction-experiment).

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
  --max-decisions 0 --max-api-calls 0 --max-replans -1 \
  --planning-context summary --boundary-review --no-sam \
  --output /outputs/2953296-full-access-transfer-ArrangeBreadBasket-001
```

Use `--sam` instead of `--no-sam` for the optional SAM condition. Keep boundary
review enabled to exercise panel confirmation. Compare with the base branch on
matched scenes across task families, including paired doors, single doors and
drawers. Report task success, regressions, recovery outcomes, operational errors
and wall time. This branch retains the older commit's execution/review schedule
but now defaults to unlimited decision/API/replan budgets; it does not import the later main-branch latency or
completion-dispute changes. Python tests and simulations must run in Thor
containers, not on the local source-work host.

Use `--max-steps 4000 --max-decisions 0 --max-api-calls 0 --max-replans -1`
to retain supervision through the 4,000-step episode horizon. Zero decision/API
caps mean unlimited; -1 means unlimited replans. The 200-call allowance in an
individual supervisor request refreshes every request and is not an episode cap.
Explicit positive decision/API caps and nonnegative replan caps remain supported.
Real execution/API failures can still end a run; unlimited supervision does not
override the bounded error handling or supply missing physical evidence.

Output-limit robustness fix: a Muse `incomplete/max_output_tokens` response now
retains its usage audit and receives at most one bounded retry per supervisor
request, with a larger output allowance (capped at 16,384) and concise-output
instructions. Retries respect the shared API budget and preserve the requested
schema, including opening-panel confirmation. If no complete response arrives,
the runner retries supervision without robot motion and reports a structured
failure when its configured failure allowance is exhausted. Partial JSON is
never treated as a decision. Rebuild both sim and supervisor for this fix.

Placement handoff is now strict: a placement or complete transfer cannot advance
until destination support, release from the gripper, and a clearly withdrawn
gripper are ALL separately observed with current camera evidence. Near-destination
overlap, open fingers alone, continued contact/holding or ambiguous/occluded
clearance cannot confirm completion. No metric or pixel-distance threshold is
invented. The object must remain at the destination while the gripper withdraws.
These structured conditions supplement the existing independent confirmation;
they do not add API calls. A simultaneous supported-grasp/completed-placement
claim is rejected. Rejected confirmations assert uncertainty, not object loss.
This applies across supported policy backends and to older placement criteria.
Models can still misread images; these checks validate the evidence contract,
not hidden simulator contact truth. Rebuild sim and supervisor for the schema.

## Verified policy-language release

The operator-requested `Open the gripper.` extension is a literal request through
`run_groot` to the frozen policy, with separately identified harness provenance.
It is not presented as upstream training wording, and no action values or forced
gripper controls are injected. It is only a temporary `controlled_release`
intervention inside a pending placement, not a standalone planned stage.

An independent review must currently observe the intended held object, correct
destination alignment, and low release height (already supported or very close
to the intended support). Unknown identity, obscured height, edge/rim alignment,
or a high drop blocks the request. No invented distance threshold is used. This
review runs even if normal boundary review is disabled and needs available API
allowance. Authorization is bound to the exact stage, current step and program,
expires on the next supervisor request, and permits at most one policy chunk.
Awomo's 16-step cadence still applies. Direct/offline unverified release requests
are rejected by the runner. After release, placement remains pending until the
usual support/release/gripper-clear checks pass. The frozen policy may still fail
to obey the language request; assess the resulting images rather than assuming
the request caused release.
