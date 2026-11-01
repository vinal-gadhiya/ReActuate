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
