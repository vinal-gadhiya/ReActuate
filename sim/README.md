# Simulation

Own RoboCasa/MuJoCo setup, environment lifecycle, rendering, simulator-side
policy clients, rollout recording, task scoring, and experiment execution.

- `Dockerfile`: Thor simulator image. Build from the repository root with
  `docker build -f sim/Dockerfile -t robocasa-thor:latest .`.
- `smoke_test.py`: physics, NVIDIA EGL, and optional kitchen checks.
- `evaluate.py`: atomic-task evaluation against the GR00T HTTP service, with
  per-episode videos and JSON results. See [the run guide](../inference/RUNNING.md).
- `run_agent.py`: atomic or composite tasks with the original instruction, a
  fixed subtask schedule, or Muse-supervised execution. See [the agent guide](../inference/AGENT.md).
- `compare.py`: sequential comparisons of GR00T, Muse, and Muse + SAM on matched
  scene seeds, with videos, traces, and aggregated task-success and latency results.

Agent episodes retain semantic object memory across decisions and replans. Muse
can request perception without motion, select a camera/target for SAM detection
or clip tracking, and request bounded fixture navigation before resuming a pending
transfer. Motion diagnostics compare declared robot measurements and rendered
images; they do not query private object poses or certify task progress.

`run_agent.py` and `compare.py` accept `--supervisor-stop-policy continue` for
simulation comparisons that should keep trying within the episode budget. A Muse
`stop` request then retries the last successfully executed `run_groot` instruction
for the current subtask for at most 16 steps, followed by another supervisor
decision on the resulting observation. Each request remains subject to the usual
shorter placement limits, control guards, and decision/API budgets. Navigation,
manual release, and instructions from prior subtasks are never replayed by this
fallback. The trace records `supervisor_stop_deferred`; results record the chosen
policy. This does not guarantee use of the full horizon or task success. The
default `halt` policy still ends immediately on a model stop request.

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
