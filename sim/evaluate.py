"""Evaluate one atomic task against a remote GR00T service."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import uuid

import numpy as np

from core.policy_http import PolicyClient


def make_policy_observation(observation, metadata, instruction=None):
    """Preserve sensors; optionally replace only the policy's language input."""
    result = {}
    for name, modality in metadata['modalities'].items():
        if name == 'action':
            continue
        for key in modality['modality_keys']:
            if key not in observation:
                raise ValueError(f'Missing observation: {key}')
            value = np.asarray(instruction if name == 'language' and instruction is not None else observation[key])
            if name == 'video':
                expected = (metadata['camera_height'], metadata['camera_width'], 3)
                if value.shape != expected or value.dtype != np.uint8:
                    raise ValueError(f'Unexpected RGB shape/type for {key}: {value.shape}/{value.dtype}')
            elif name == 'state':
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError(f'Invalid state: {key}')
            elif name == 'language' and (value.ndim != 0 or not str(value.item()).strip()):
                raise ValueError('Task instruction is empty or malformed')
            result[key] = value[None]  # One temporal observation, no batch axis.
    return result


def validate_actions(actions, action_space, horizon):
    if set(actions) != set(action_space.spaces):
        raise ValueError(f'Action keys differ: {set(actions) ^ set(action_space.spaces)}')
    for key, space in action_space.spaces.items():
        values = np.asarray(actions[key])
        expected = (horizon, *space.shape)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f'Invalid action chunk for {key}: expected finite {expected}, got {values.shape}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-url', default='http://inference:8000')
    parser.add_argument('--task', default='OpenCabinet')
    parser.add_argument('--split', choices=('pretrain', 'target'), default='pretrain')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--action-steps', type=int, default=16)
    parser.add_argument('--max-steps', type=int, help='Default: RoboCasa task horizon')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', type=Path, default=Path('/outputs/gr00t-open-cabinet'))
    args = parser.parse_args()
    if args.episodes <= 0 or args.action_steps <= 0 or args.timeout <= 0:
        parser.error('Episodes, action steps, and timeout must be positive')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')

    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa  # Registers tasks.
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if args.task not in ATOMIC_TASK_DATASETS:
        parser.error(f'Task must be in ATOMIC_TASK_DATASETS, got {args.task}')
    max_steps = args.max_steps or get_task_horizon(args.task)
    client = PolicyClient(args.policy_url, timeout=args.timeout)
    metadata = client.metadata()
    if metadata['data_config'] != 'panda_omron' or metadata['embodiment'] != 'PandaOmron':
        raise ValueError('Expected a PandaOmron policy')
    if args.action_steps > metadata['action_horizon']:
        raise ValueError('--action-steps exceeds the policy action horizon')

    # A run directory is never overwritten, including partial failed runs.
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['output'] = str(args.output)
    config['resolved_max_steps'] = max_steps
    config['policy'] = metadata
    config['versions'] = {key: version(key) for key in ('robocasa', 'robosuite', 'mujoco', 'numpy', 'gymnasium')}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    run_id = uuid.uuid4().hex
    results = []
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            env = None
            started = time.monotonic()
            inference_seconds = 0.0
            steps = 0
            success = False
            video_path = args.output / f'episode-{episode:03d}.mp4'
            try:
                env = gym.make(
                    f'robocasa/{args.task}', robots='PandaOmron', split=args.split,
                    seed=episode_seed,
                    camera_widths=metadata['camera_width'],
                    camera_heights=metadata['camera_height'],
                )
                observation, _ = env.reset(seed=episode_seed)
                camera_key = metadata['modalities']['video']['modality_keys'][0]
                fps = int(env.unwrapped.env.control_freq)
                with imageio.get_writer(video_path, fps=fps) as video:
                    video.append_data(observation[camera_key])
                    while steps < max_steps and not success:
                        request_id = f'{run_id}:{episode}:{steps}'
                        inputs = make_policy_observation(observation, metadata)
                        inputs['__request_id'] = np.asarray(request_id)
                        before = time.monotonic()
                        actions = client.predict(inputs)
                        inference_seconds += time.monotonic() - before
                        if actions.pop('__request_id').item() != request_id:
                            raise ValueError('Stale/mismatched policy response')
                        validate_actions(actions, env.action_space, metadata['action_horizon'])
                        for index in range(min(args.action_steps, max_steps - steps)):
                            action = {key: value[index].copy() for key, value in actions.items()}
                            observation, _, terminated, truncated, info = env.step(action)
                            steps += 1
                            video.append_data(observation[camera_key])
                            success = bool(info.get('success', False))
                            if success or terminated or truncated:
                                break
                        if terminated or truncated:
                            break
                result = {
                    'episode': episode, 'seed': episode_seed, 'success': success,
                    'steps': steps, 'wall_seconds': time.monotonic() - started,
                    'inference_seconds': inference_seconds, 'video': video_path.name,
                    'instruction': str(observation['annotation.human.task_description']),
                }
                results.append(result)
                with (args.output / 'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                print(json.dumps(result), flush=True)
            finally:
                if env is not None:
                    env.close()
    except Exception as error:
        (args.output / 'error.json').write_text(json.dumps({
            'error': str(error), 'completed_episodes': len(results),
        }, indent=2) + '\n')
        raise
    summary = {
        'task': args.task, 'split': args.split, 'episodes': len(results),
        'successes': sum(item['success'] for item in results),
        'success_rate': sum(item['success'] for item in results) / len(results),
        'max_steps': max_steps, 'action_steps': args.action_steps,
        'evaluation': 'single-task diagnostic; not a leaderboard reproduction',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
