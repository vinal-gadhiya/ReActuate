# Simulation

Own RoboCasa/MuJoCo setup, environment lifecycle, rendering, simulator-side
policy clients, rollout recording, task scoring, and experiment execution.

- `Dockerfile`: Thor simulator image. Build from the repository root with
  `docker build -f sim/Dockerfile -t robocasa-thor:latest .`.
- `smoke_test.py`: physics, NVIDIA EGL, and optional kitchen checks.
- `evaluate.py`: atomic-task evaluation against the GR00T HTTP service, with
  per-episode videos and JSON results. See [the run guide](../inference/RUNNING.md).

Assets use `robocasa-thor-assets:/opt/robocasa-assets`; generated artifacts go
under ignored `outputs/`. Keep model weights and API credentials out of this image.

The simulation process owns the authoritative episode, step, and instruction
version. Send observations to inference and validate returned actions before
applying them. Start synchronously at action-chunk boundaries: pause simulated
time during inference/API calls and measure wall-clock latency separately.
Reject responses whose episode, observation, or instruction is obsolete.

Privileged task checks and MuJoCo state belong in evaluation/label collection.
Expose only declared sensors and proprioception to the normal controller.
Use a separately named oracle experiment for privileged controller inputs.

The evaluation client uses the shared NumPy HTTP interface, keeping torch and
the GR00T package out of the simulator image. RoboCasa's Gym wrapper handles the
PandaOmron observation/action mapping; the inference service applies the
checkpoint's own policy transforms. For the upstream random-rollout helper, pass
`env.unwrapped` to avoid Gymnasium's `OrderEnforcing` attribute-forwarding error.
