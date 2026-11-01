# GR00T N1.5 on Thor

Run every command below on Jetson AGX Thor. Installs and Python execution happen
inside Docker containers. The host needs Docker Compose 2.30+ with NVIDIA
Container Toolkit and a CUDA 13-compatible driver.

## 1. Build the images

From the repository root:

```bash
docker compose build inference sim
```

The inference image uses Python 3.12, PyTorch 2.9.1 / torchvision 0.24.1 from
the official CUDA 13 ARM64 wheel index, and the leaderboard's pinned GR00T fork.
The simulator keeps its own Python environment. The build runs dependency and
GR00T import checks without requiring a GPU.

Source checkout downloads LFS objects only under `gr00t/`. Upstream demo media
is excluded so missing media objects do not block the inference build.

The fork's training dependency list is replaced with the inference requirements
in `inference/requirements.txt`. Its bundled Eagle config and vision constructor
use eager attention to avoid a FlashAttention extension build. Weights and policy transforms remain
unchanged; this backend/version combination needs validation on Thor before
performance or benchmark comparisons. Exact transitive packages and base image
digests are not fully locked; resolved packages are saved in `/opt/build-info`.

The CUDA 13 cuSPARSELt 0.8.0 ARM64 wheel has an incorrect internal `sbsa`
platform tag. The build repairs that metadata only after checking its version,
ELF architecture, installed file hashes, and library loading. It updates the
metadata's RECORD entry and then runs `pip check` normally. Package versions and
native binaries are unchanged. The repair is recorded in
`/opt/build-info/cusparselt-metadata.json`.

The checkpoint uses SigLIP vision. Optional RADIO imports are deferred until a
RADIO backbone is selected, preventing its FlashAttention dependency from loading
for SigLIP. Build checks load Eagle's dynamic configuration and model class
without allocating weights or requiring a GPU. Generated Transformers modules
use a container-local cache; downloaded model artifacts remain persistent.

## 2. Download the checkpoint

```bash
docker compose run --rm --no-deps inference \
  python -m inference.download_checkpoint
```

Downloads the two weight shards, configuration, and `experiment_cfg` metadata
from the RoboCasa checkpoint repository. Optimizer and trainer state are
excluded. Allow at least 10 GB for checkpoint storage, plus image/cache space.
The downloader resolves the requested revision to a commit and records it in
`download_manifest.json`. Use `--revision <commit>` for subsequent identical
downloads. No training datasets are needed.

Weights persist in the named volume `objectworld-checkpoints`, at:

```text
/checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000
```

## 3. Start inference

```bash
docker compose up -d inference
docker compose logs -f inference
```

Wait for `Policy ready`. First startup loads the weights and may populate the
Hugging Face cache. The server checks CUDA availability, BF16 support, and a real
GPU matrix multiplication before loading GR00T. It uses `panda_omron`,
`new_embodiment`, and four denoising steps, matching the benchmark defaults.
Startup errors are fatal; the server never falls back silently to CPU.

Check readiness from a container:

```bash
docker compose exec inference python -c \
  'from urllib.request import urlopen; print(urlopen("http://127.0.0.1:8000/health").read().decode())'
```

The service is available to the simulator as `http://inference:8000`. No port is
published on the host. Model inference and rendering share Thor's GPU resources.

## 4. Prepare simulator assets

If `robocasa-thor-assets` already contains downloaded assets, skip this step.

```bash
docker compose run --rm --no-deps sim \
  python -m robocasa.scripts.download_kitchen_assets
```

Answer `y` and check the downloader output for failures. This reuses the same
named asset volume as the standalone simulator setup.

## 5. Evaluate a simple atomic task

Run one `OpenCabinet` episode first:

```bash
docker compose run --rm sim \
  python -m sim.evaluate \
    --task OpenCabinet \
    --split pretrain \
    --episodes 1 \
    --seed 0 \
    --output /outputs/gr00t-open-cabinet-001
```

Each inference request uses the current three RGB cameras, proprioception, and
the environment's task instruction. The server applies the checkpoint's
normalization and transforms. The simulator executes up to 16 predicted actions
before asking for another chunk, stopping at success or the task horizon.
Images retain the RoboCasa Gym wrapper's orientation; do not flip them again.
Physics pauses during inference. Base actions are left intact, unlike the
random-rollout helper, which zeros them.

The run writes to `outputs/gr00t-open-cabinet-001` on the Thor host:

- `episode-000.mp4`: first policy camera, recorded at the simulator control rate.
- `episodes.jsonl`: success, seed, steps, instruction, and timing per episode.
- `config.json`: task settings, server/checkpoint metadata, and package versions.
- `summary.json`: overall success rate.
- `error.json`: failure details if a run aborts; failed runs have no success summary.

Output directories must be new so runs cannot overwrite one another. A failed
task is a normal evaluation outcome; transport/model errors abort the run.

Then run a small batch with a new output path:

```bash
docker compose run --rm sim \
  python -m sim.evaluate \
    --task OpenCabinet --split pretrain --episodes 10 --seed 0 \
    --output /outputs/gr00t-open-cabinet-010
```

`PickPlaceCounterToCabinet` is another atomic task to try. `--max-steps` overrides
the registry's task horizon for debugging, and `--action-steps` changes execution
chunk length. These are diagnostic single-task evaluations, not leaderboard
reproductions. The checkpoint is stochastic: environment seeds are recorded and
the server's random stream is seeded at startup; restart the server for the same
initial policy RNG state. Bitwise deterministic GPU inference is not guaranteed.

## Container-only tests

Muse output-limit failures (`incomplete`, `max_output_tokens`) get at most one
retry per supervisor request, within the shared API-call budget. The retry doubles
the configured output allowance up to 16,384 tokens and requests concise JSON.
Truncated text is never executed; its available token usage is retained in the
audit. If no retry fits or it fails, the runner requests a fresh decision without
policy motion and stops cleanly if the supervisor failure limit is exceeded.

Run the lightweight protocol/adapter tests with the source mounted read-only:

```bash
docker compose run --rm --no-deps \
  -v "$(pwd):/workspace:ro" -w /workspace \
  inference python -m unittest discover -s tests -v
```

These tests use a fake policy to verify HTTP round trips, temporal observation
shapes, image preservation, action validation, and the packaging patch. They do
not download weights or require GPU inference. The first real episode validates
checkpoint loading, rendering, and end-to-end action execution on Thor.

## Stop

```bash
docker compose down
```

Named checkpoint, asset, and cache volumes persist. Avoid `docker compose down -v`
unless you intend to delete them.

## References

- [RoboCasa GR00T submission](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/gr00t_n1.5_2026_05_19.md)
- [Pinned policy implementation](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/model/policy.py)
- [Checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000)
- [Official PyTorch CUDA 13 wheels](https://download.pytorch.org/whl/cu130/torch/)
