# Awomo-0.5I on the existing harness

This adds an optional `awomo-0.5i` backend to
`experiment/2953296-full-access-transfer`. Select `compose.awomo.yaml` instead
of the PatchWAM overlay on every command. Muse, optional SAM, original goal
tracking, recovery planning, opening verification and output artifacts use the
same harness. GR00T and PatchWAM remain selectable through their existing images.

## What is pinned and preserved

- Official source: Awomo-WestlakeDI/Awomo-0.5,
  `b11d82f7ee198955d0e9d26b3ebeed1dafea03b4`.
- Released RoboCasa checkpoint: `Auwomo/Awomo-0.5I-RoboCasa365`,
  revision `d02e3560a9e5e649bb0718a97b9e4b923df59594`. Both checkpoint files are
  checked against the released SHA256SUMS values. Base models use the exact
  revisions in the checkpoint configuration; the downloader records local hashes.
- Three raw RGB cameras at 256×256, actual observations every four control steps,
  and predictions every 16 steps. The official policy handles H264 CRF23,
  wrist/external canvas composition, state16→state80 conversion, nine historical
  slots spaced 40 steps apart, normalization, 16 denoising steps and action output.
- Official output clipping, mode/gripper discretization and torso behavior are
  preserved. The bridge only rearranges official `convert_action`-order rows into
  the simulator's named action fields. It applies no additional action overrides.
- Episode-scoped history uses a fresh UUID per episode. A new episode clears the
  upstream episode and prompt-token cache. Missing frames, stale responses,
  interleaved sessions, invalid shapes or nonfinite actions fail explicitly.

One necessary harness adaptation: the upstream policy retains its first task
instruction. Before each subsequent prediction our bridge updates that retained
instruction to Muse's latest explicit instruction, keeping visual history. Model
weights, preprocessing and sampling stay unchanged. Baseline runs keep the
original task instruction; no prior action tail survives an instruction change.

## Relational instruction experiment

`sim.run_agent` now defaults to `--instruction-mode auto`: Awomo uses relational
language, while GR00T/PatchWAM retain atomic grounding. `--instruction-mode atomic`
restores the strict catalog/decomposed-plan setting for an ablation;
`--instruction-mode relational` explicitly selects the Awomo experiment.

Muse can use observed appearance and spatial anchors in basic action sentences,
for example `Place the mug near the brown mug.` or `Take the mug from the cabinet
beside the sink and place it beside the brown mug.` These are illustrations,
not claims about a scene. Muse must identify the target/reference/destination
from supplied observations, preserve original goal constraints and clarify
ambiguous camera-relative left/right. An anchor object is not automatically the
support surface. Combined transfers are allowed in this mode when helpful;
one executable tool call per decision still applies. Instructions pass through
verbatim. Non-catalog frames are logged as a harness language experiment, not
upstream training annotations.

Current evidence-backed `reference_refinement` may clarify the same pending
object/destination without claiming execution failure. It is restricted to
relational mode. Original requirements, per-panel opening confirmation, low-height
release permits, strict release/gripper-clear completion, and Awomo's 16-step
cadence remain enforced. Arbitrary action chains and hand-release synonyms are
not accepted. `Open the gripper.` still requires the separate release gate.

After committing/pushing, rebuild sim/supervisor and rerun their tests using the
commands below. Add `--instruction-mode relational` to the Muse rollout command,
or use `--instruction-mode atomic` on the same scene/seed for the comparison.
Language flexibility is an experiment; mock contract checks do not establish
improved grounding or task success.

The model requires 16-step action chunks. Use `--action-steps 16` and
`--decision-steps 32` (the Awomo default), or another multiple of 16 up to 80.
Generated intervals that violate the cadence are rejected before motion. The
last physical episode remainder may be shorter, with its unused action tail
discarded. Do not run multiple simulators against one stateful inference server.
The overlay allows 600 seconds per policy prediction for Thor warmup/inference;
`--policy-timeout` overrides it in `run_agent` (`--timeout` in `evaluate`). This
is a request timeout, not an episode wall-clock deadline.

## Thor build, assets and checks

Commit/push these changes, then update this branch on Thor. All Python execution,
builds, package installation and simulation belong in Thor containers.

```bash
git fetch origin
git switch experiment/2953296-full-access-transfer
git pull --ff-only

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent build inference sim supervisor

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent \
  run --rm --no-deps supervisor python -m unittest discover -s tests -v

sudo docker compose -f compose.yaml -f compose.awomo.yaml \
  run --rm --no-deps inference python -m inference.download_awomo

sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent \
  up -d --no-deps --force-recreate --wait --wait-timeout 1800 inference supervisor

sudo docker compose -f compose.yaml -f compose.awomo.yaml \
  logs --tail=100 inference
```

Supply `HF_TOKEN` in the existing Compose environment if needed to access the
checkpoint's base models. Downloads use a separate `reactuate-awomo-05i-checkpoints`
volume; existing checkpoint volumes are retained.

The inference image targets ARM64 Thor using CUDA 13, Python 3.12,
PyTorch 2.9.1 and torchvision 0.24.1, with the other official image-inference
dependency versions pinned. Upstream evaluated Python 3.11/PyTorch 2.8/CUDA 12.8
on RTX GPUs. This Thor port needs build/runtime validation; it has not been GPU
tested in the local source workspace. Startup requires CUDA BF16, runs a real
GPU kernel, checks all model parameters reside on CUDA and logs GPU, Torch, CUDA,
checkpoint provenance, compression, scheduler and history settings. There is no
CPU inference fallback. Missing/unsupported assets fail rather than substitute a
different model. The NumPy/mock tests verify wiring, not task quality.

## Smoke run, then baseline and Muse

First verify a short atomic rollout can infer, move and save artifacts:

```bash
sudo docker compose -f compose.yaml -f compose.awomo.yaml run --rm sim \
  python -m sim.evaluate --task CloseBlenderLid --split pretrain --seed 0 \
  --episodes 1 --expected-policy awomo-0.5i --action-steps 16 --max-steps 64 \
  --output /outputs/awomo-close-blender-smoke-001
```

Then run the focused composite-unseen baseline:

```bash
sudo docker compose -f compose.yaml -f compose.awomo.yaml run --rm sim \
  python -m sim.run_agent --task ArrangeBreadBasket --mode baseline \
  --split pretrain --seed 0 --episodes 1 --expected-policy awomo-0.5i \
  --require-unseen-composite --action-steps 16 --max-steps 4000 \
  --output /outputs/awomo-ArrangeBreadBasket-baseline-001
```

Run Muse on the same recorded scene:

```bash
sudo docker compose -f compose.yaml -f compose.awomo.yaml --profile agent run --rm sim \
  python -m sim.run_agent --task ArrangeBreadBasket --mode agent \
  --split pretrain --seed 0 --episodes 1 --expected-policy awomo-0.5i \
  --require-unseen-composite --execution-policy episode \
  --action-steps 16 --decision-steps 32 --max-steps 4000 \
  --max-decisions 0 --max-api-calls 0 --max-replans -1 \
  --planning-context summary --boundary-review --no-sam \
  --reference-scene /outputs/awomo-ArrangeBreadBasket-baseline-001/episode-000/scene.json \
  --output /outputs/awomo-ArrangeBreadBasket-muse-001
```

Replace `--no-sam` with `--sam` for optional visual grounding. Composite runs save
videos, exact observation images, traces, API audits, plans, goal history, scene
identity and timing in the usual output directories. Select another unseen task
by changing `--task` and using a fresh output directory.

`sim.compare` also supports `--expected-policy awomo-0.5i`, including baseline,
Muse and Muse+SAM conditions, with `--action-steps 16 --decision-steps 32` and
the unlimited supervision flags above. Compare scene-matched success, recovery
quality, operational errors, physical steps and wall time across tasks/seeds.
Changed instruction handling is an agent adaptation; do not claim that these
results reproduce upstream's 1,500-episode evaluation or guarantee its task scores.

References: [official RoboCasa evaluation](https://github.com/Awomo-WestlakeDI/Awomo-0.5/tree/b11d82f7ee198955d0e9d26b3ebeed1dafea03b4/eval_robocasa365),
[released weights](https://huggingface.co/Auwomo/Awomo-0.5I-RoboCasa365/tree/d02e3560a9e5e649bb0718a97b9e4b923df59594).
