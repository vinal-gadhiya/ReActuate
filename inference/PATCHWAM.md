# PatchWAM backend on Thor

This adds a separate frozen policy backend. The default Compose file, GR00T
server, GR00T dependency set and GR00T checkpoint downloader are unchanged.
Select PatchWAM with `-f compose.yaml -f compose.patchwam.yaml` on **every**
Compose command below. The overlay replaces only the inference service image;
it does not run GR00T alongside PatchWAM. Existing GR00T checkpoints remain in
their original volume, mounted read-only. PatchWAM has a separate volume.

## Release and adapter

- Source: [PatchWAM-RoboCasa](https://github.com/TianhengW/PatchWAM-RoboCasa),
  commit `7783ae2447ef26c10b2b0470753dfad6be85202e`.
- Checkpoint: [QZWang/robocasa-human300-step559660](https://huggingface.co/QZWang/robocasa-human300-step559660),
  revision `9df2fa292952d4d10ef61083c732862c880b1d07`, file `step_559660.pt`.
- Required SHA256: `7877dd2546d7e2fe489aab79f84476266724b7392d48fe2f2861e9752cd4dccb`.
- Base assets: FLUX.2-klein-base-4B transformer, FLUX.2-dev autoencoder, Qwen3-4B.
  HF revisions are resolved once and saved with file checksums in `manifest.json`.
  Repeating the downloader reuses saved revisions unless explicitly overridden.

The adapter invokes the released `APPolicy` directly, preserving its three-camera
compact 288x256 layout, prompt template, z-score normalization, and action
denormalization. Inputs are current 256x256 camera frames plus 16 state values
in the dataset's order: base position, base quaternion, relative end-effector
position, relative end-effector quaternion, gripper positions. The 12 predicted
action values are split by their named dataset fields. There is no action
clipping, forced gripper command, or geometry controller in this adapter.

The wire protocol is the existing NumPy `/metadata` and `/predict` HTTP protocol.
`data_config=panda_omron` describes that interface; `policy=patchwam` identifies
the model. Muse receives PatchWAM-specific executor context. `run_groot(...)`
remains the compatibility tool name in traces/programs, but calls PatchWAM when
this backend is selected. SAM and Muse stay in the separate supervisor container.

## Build and CPU tests

Run these commands on Thor from the repository root. No local laptop builds or
training are needed. PatchWAM uses its own dependencies and CUDA 13 PyTorch image.
The image applies the same checked cuSPARSELt 0.8.0 ARM64 wheel-metadata repair
as the GR00T image before `pip check`. It verifies the AArch64 binary and its
installed hashes, then corrects only the wheel platform tag and RECORD metadata.
The released BF16 implementation requires a compatible GPU; this setup does not
provide an RTX 2080 Ti/FP16 conversion or CPU fallback.

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  build inference sim supervisor

sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm --no-deps supervisor python -m unittest discover -s tests -v
```

Tests cover the adapter's state/action ordering, preserved pixels and actions,
invalid observations, HTTP request IDs and instruction changes, plus existing
GR00T regression tests. They do not load real model weights.

## Download weights

Some base-model repositories may require accepting their access conditions on
Hugging Face. Set `HF_TOKEN` in your local `.env` if needed; do not commit it.
This downloads substantial model files into the PatchWAM volume, not the repo.

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml \
  run --rm --no-deps inference python -m inference.download_patchwam

sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  up -d --wait --wait-timeout 900 inference supervisor

sudo docker compose -f compose.yaml -f compose.patchwam.yaml \
  logs --tail=100 inference
```

Startup validates the checkpoint and base-asset hashes, tests a CUDA BF16 kernel,
and loads the upstream model. The service becomes healthy only after loading.
Default inference uses 10 denoising steps and global seed 7. The seed applies to
the server process; it is not reset at each prediction or episode. Model metadata
records the source, resolved asset revisions, configuration/statistics hashes,
seed, denoising settings, and GPU/software versions.

## First run: unchanged original instruction

Use the native **16-action** execution interval first. This baseline runs no
Muse/SAM requests and uses the same simulator harness as later comparisons.
Start with one atomic task before the composite:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml run --rm sim \
  python -m sim.run_agent \
  --task CloseFridge --mode baseline --split pretrain --seed 0 \
  --action-steps 16 --decision-steps 64 --max-steps 900 \
  --output /outputs/patchwam-close-fridge-001
```

Then run the composite original-instruction baseline:

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml run --rm sim \
  python -m sim.run_agent \
  --task ArrangeTea --mode baseline --split pretrain --seed 0 \
  --require-unseen-composite --action-steps 16 --decision-steps 64 \
  --max-steps 2250 --output /outputs/patchwam-tea-baseline-001
```

This is a diagnostic, not a reproduction of the published aggregate score.
Validate atomic skills and the original-instruction baseline before attributing
any difference to Muse. The published reference uses RoboCasa 1.0.1,
`split=pretrain`, 16-step replanning, 10 denoising steps, and 50 trials per task.
Our explicit scene selection/reset procedure and policy RNG history must also
be accounted for when interpreting differences from the release's evaluator.

## Muse and optional SAM

This verifies the same starting scene against the baseline above, without
executing the baseline again. Use `--sam` in place of `--no-sam` to add grounding.

```bash
sudo docker compose -f compose.yaml -f compose.patchwam.yaml --profile agent \
  run --rm sim python -m sim.run_agent \
  --task ArrangeTea --mode agent --split pretrain --seed 0 --no-sam \
  --require-unseen-composite --action-steps 16 --decision-steps 64 \
  --supervisor-stop-policy continue --max-supervisor-failures 3 \
  --max-decisions 190 --max-api-calls 200 --max-steps 2250 \
  --reference-scene /outputs/patchwam-tea-baseline-001/episode-000/scene.json \
  --output /outputs/patchwam-tea-muse-001
```

For paired comparisons, the same overlay works with `python -m sim.compare`:
specify `--action-steps 16 --decision-steps 64 --conditions baseline muse`.
All conditions use PatchWAM; the comparison's `baseline` means the selected
policy with the original task instruction. Healthy supervision may still request
shorter intervals or change instructions. Tool boundaries discard chunk tails;
those effects must be measured separately from any planning benefit.

## Return to GR00T

```bash
sudo docker compose --profile agent up -d --wait inference supervisor
```

Omitting the overlay restores the original GR00T inference service definition.
Use the normal GR00T commands and its execution settings afterward. Do not mix a
GR00T baseline with a PatchWAM supervised run and call that a planner ablation.

## Validation status

Source/protocol integration is implemented. GPU loading, dependency installation,
memory use, latency and actual task execution require validation on Thor. No
PatchWAM rollout success or reproduction result is claimed yet.
