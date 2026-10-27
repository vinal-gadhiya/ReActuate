"""Run a composite task with baseline, fixed-plan, or Muse-supervised control."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import uuid

import numpy as np

from core.agent import parse_program, validate_decision, validate_plan
from core.policy_http import PolicyClient, SupervisorClient
from sim.evaluate import make_policy_observation, validate_actions


def gripper_supported(env):
    controller = env.unwrapped.env.robots[0].composite_controller.part_controllers.get('right')
    if controller is None:
        return False
    return (
        getattr(controller, 'name', None) == 'OSC_POSE'
        and getattr(controller, 'input_type', None) == 'delta'
        and getattr(controller, 'impedance_mode', None) == 'fixed'
        and np.allclose(controller.input_min, -controller.input_max)
        and np.allclose(controller.output_min, -controller.output_max)
    )


class EpisodeTools:
    def __init__(self, env, observation, policy, metadata, video, max_steps, chunk_steps, allow_gripper=False):
        self.env, self.observation, self.policy, self.metadata, self.video = env, observation, policy, metadata, video
        self.max_steps, self.chunk_steps = max_steps, chunk_steps
        self.steps, self.success, self.ended = 0, False, False
        self.inference_seconds = 0.0
        self.run_id = uuid.uuid4().hex
        self.allow_gripper = allow_gripper and gripper_supported(env)
        self.camera = metadata['modalities']['video']['modality_keys'][0]

    @property
    def done(self):
        return self.success or self.ended or self.steps >= self.max_steps

    def step(self, action):
        if self.done:
            return
        self.observation, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.success = bool(info.get('success', False))
        self.ended = bool(terminated or truncated)
        self.video.append_data(self.observation[self.camera])

    def run_groot(self, instruction, steps):
        limit = min(self.steps + steps, self.max_steps)
        while self.steps < limit and not self.done:
            inputs = make_policy_observation(self.observation, self.metadata, instruction=instruction)
            request_id = f'{self.run_id}:{self.steps}'
            inputs['__request_id'] = np.asarray(request_id)
            started = time.monotonic()
            actions = self.policy.predict(inputs)
            self.inference_seconds += time.monotonic() - started
            if actions.pop('__request_id').item() != request_id:
                raise ValueError('Mismatched policy response')
            validate_actions(actions, self.env.action_space, self.metadata['action_horizon'])
            for index in range(min(self.chunk_steps, limit - self.steps)):
                self.step({key: value[index].copy() for key, value in actions.items()})
                if self.done:
                    break
        # No queued actions survive this call or an instruction transition.

    def set_gripper(self, closed, steps):
        if not self.allow_gripper:
            raise ValueError('Gripper primitive requires an explicitly enabled, validated delta OSC controller')
        expected = {
            'action.end_effector_position': (3,), 'action.end_effector_rotation': (3,),
            'action.gripper_close': (1,), 'action.base_motion': (4,), 'action.control_mode': (1,),
        }
        if set(self.env.action_space.spaces) != set(expected):
            raise ValueError('Unknown PandaOmron action schema')
        for key, shape in expected.items():
            if self.env.action_space.spaces[key].shape != shape:
                raise ValueError(f'Unknown action dimensions: {key}')
        for _ in range(steps):
            if self.done:
                break
            action = {key: np.zeros(shape, dtype=np.float32) for key, shape in expected.items()}
            action['action.gripper_close'][:] = float(closed)
            # Gym maps control_mode < .5 to arm control (-1); base motion stays zero.
            self.step(action)


def run_subtasks(tools, plan, mode, decide, record, decision_steps=40, max_decisions=40, max_replans=2):
    index, spent, decisions, replans = 0, 0, 0, 0
    while not tools.done and index < len(plan):
        step = plan[index]
        remaining = min(step['max_steps'] - spent, tools.max_steps - tools.steps)
        if mode != 'agent':
            if remaining <= 0:
                record({'event': 'scheduled_transition', 'subtask': step['id'], 'verified': False, 'step': tools.steps})
                index, spent = index + 1, 0
                continue
            decision = {
                'status': 'continue', 'evidence': 'Fixed execution schedule; no visual verification.',
                'program': f'run_groot(instruction={step["instruction"]!r}, steps={min(decision_steps, remaining)})',
                'replacement_steps': [],
            }
        else:
            if decisions >= max_decisions:
                return 'decision_budget'
            decision = decide(plan, index, spent, max(0, min(decision_steps, remaining)))
            decisions += 1
            if decision is None:
                return 'api_budget'
            validate_decision(decision, max(0, min(decision_steps, remaining)), tools.allow_gripper)
        record({'event': 'decision', 'subtask': step['id'], 'step': tools.steps, **decision})
        status = decision['status']
        if status == 'stop':
            return 'supervisor_stop'
        if status == 'complete':
            index, spent = index + 1, 0
            continue
        if status == 'replan':
            if replans >= max_replans:
                return 'replan_budget'
            replacement = decision['replacement_steps']
            completed_ids = {item['id'] for item in plan[:index]}
            if completed_ids & {item['id'] for item in replacement}:
                raise ValueError('Replan reused a completed subtask ID')
            plan = plan[:index] + replacement
            replans, spent = replans + 1, 0
            continue
        for name, arguments in parse_program(decision['program'], min(decision_steps, remaining), tools.allow_gripper):
            before = tools.steps
            getattr(tools, name)(**arguments)
            spent += tools.steps - before
            record({'event': 'tool', 'subtask': step['id'], 'tool': name, 'arguments': arguments,
                    'step_before': before, 'step_after': tools.steps})
            if tools.done:
                break
    if tools.success:
        return 'task_success'
    if tools.steps >= tools.max_steps:
        return 'episode_step_budget'
    if tools.ended:
        return 'environment_ended'
    return 'plan_exhausted_without_task_success'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='MakeIceLemonade')
    parser.add_argument('--mode', choices=('baseline', 'fixed', 'agent'), default='agent')
    parser.add_argument('--plan', type=Path, help='JSON plan for fixed mode or a controlled agent starting plan')
    parser.add_argument('--policy-url', default='http://inference:8000')
    parser.add_argument('--supervisor-url', default='http://supervisor:8001')
    parser.add_argument('--split', choices=('pretrain', 'target'), default='pretrain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--action-steps', type=int, default=8)
    parser.add_argument('--decision-steps', type=int, default=40)
    parser.add_argument('--max-decisions', type=int, default=40)
    parser.add_argument('--max-api-calls', type=int, default=60)
    parser.add_argument('--max-replans', type=int, default=2)
    parser.add_argument('--allow-gripper', action='store_true', help='Enable the guarded delta-OSC gripper primitive')
    parser.add_argument('--plan-only', action='store_true', help='Generate and save a plan without stepping the robot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('episodes', 'action_steps', 'decision_steps', 'max_decisions', 'max_api_calls'):
        if getattr(args, key) <= 0:
            parser.error(f'{key} must be positive')
    if args.decision_steps > 80 or args.max_api_calls > 200 or args.max_replans < 0:
        parser.error('Decision steps <= 80, API calls <= 200, and replans >= 0 are required')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    if args.mode == 'fixed' and args.plan is None:
        parser.error('fixed mode requires --plan')
    if args.mode == 'baseline' and args.plan is not None:
        parser.error('baseline mode always uses the original task instruction')
    if args.plan_only and args.mode != 'agent':
        parser.error('--plan-only requires agent mode')
    starting_plan = None
    if args.plan:
        starting_plan = validate_plan(json.loads(args.plan.read_text()))

    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if args.task not in ATOMIC_TASK_DATASETS and args.task not in COMPOSITE_TASK_DATASETS:
        parser.error('Unknown registered RoboCasa task')
    max_steps = args.max_steps or get_task_horizon(args.task)
    policy = PolicyClient(args.policy_url)
    metadata = policy.metadata()
    if metadata.get('data_config') != 'panda_omron' or metadata.get('embodiment') != 'PandaOmron':
        raise ValueError('A PandaOmron policy is required')
    if args.action_steps > metadata['action_horizon']:
        parser.error('Action steps exceed policy horizon')
    supervisor = SupervisorClient(args.supervisor_url, timeout=300) if args.mode == 'agent' else None
    supervisor_metadata = supervisor.metadata() if supervisor else None
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'resolved_max_steps': max_steps, 'policy': metadata, 'supervisor': supervisor_metadata,
                   'starting_plan': starting_plan,
                   'versions': {key: version(key) for key in ('robocasa', 'robosuite', 'mujoco', 'numpy')}})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    for episode in range(args.episodes):
        directory = args.output / f'episode-{episode:03d}'
        directory.mkdir()
        env = None
        started = time.monotonic()
        history = []

        def record(event):
            if event['event'] == 'decision' and event.get('program'):
                filename = f'program-{len(history):03d}.py'
                (directory / filename).write_text(event['program'] + '\n')
                event = {**event, 'program_path': filename}
            with (directory / 'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            if event['event'] in ('decision', 'tool', 'scheduled_transition'):
                history.append(event)
                print(json.dumps(event), flush=True)

        try:
            env = gym.make(
                f'robocasa/{args.task}', robots='PandaOmron', split=args.split, seed=args.seed + episode,
                camera_widths=metadata['camera_width'], camera_heights=metadata['camera_height'],
            )
            observation, _ = env.reset(seed=args.seed + episode)
            goal = str(observation['annotation.human.task_description'])
            camera_keys = metadata['modalities']['video']['modality_keys']
            api_calls, request_number = 0, 0
            previous_images = None

            def ask(operation, context):
                nonlocal api_calls, request_number, previous_images
                if api_calls >= args.max_api_calls:
                    return None
                current = {f'current:{key}': tools.observation[key].copy() for key in camera_keys}
                images = dict(current)
                if previous_images is not None and operation == 'decide':
                    images.update({key.replace('current:', 'before:', 1): frame for key, frame in previous_images.items()})
                references = []
                for number, (label, frame) in enumerate(sorted(images.items())):
                    filename = f'observation-{request_number:03d}-{number}.jpg'
                    imageio.imwrite(directory / filename, frame)
                    references.append({'label': label, 'path': filename})
                context.update({'original_goal': goal, 'episode_steps_left': max_steps - tools.steps,
                                'api_calls_left': args.max_api_calls - api_calls})
                record({'event': 'supervisor_request', 'operation': operation, 'request': request_number,
                        'step': tools.steps, 'context': context, 'images': references})
                response = supervisor.ask(operation, context, images)
                api_calls += response['api_calls']
                record({'event': 'supervisor_response', 'request': request_number, **response})
                previous_images = current
                request_number += 1
                return response['result']

            with imageio.get_writer(directory / 'rollout.mp4', fps=int(env.unwrapped.env.control_freq)) as video:
                video.append_data(observation[camera_keys[0]])
                tools = EpisodeTools(env, observation, policy, metadata, video, max_steps, args.action_steps, args.allow_gripper)
                if args.mode == 'baseline':
                    plan = [{'id': 'original', 'instruction': goal, 'target': 'task objects',
                             'success_criteria': 'Complete the original task', 'max_steps': max_steps}]
                elif starting_plan is not None:
                    plan = [dict(step) for step in starting_plan]
                else:
                    plan = validate_plan(ask('plan', {'episode_horizon': max_steps}))
                record({'event': 'initial_plan', 'goal': goal, 'steps': plan})
                (directory / 'plan.json').write_text(json.dumps({'steps': plan}, indent=2) + '\n')

                def decide(active_plan, index, spent, limit):
                    return ask('decide', {
                        'step': active_plan[index], 'subtask_steps_used': spent,
                        'completed_steps': active_plan[:index], 'remaining_steps': active_plan[index + 1:],
                        'history': history[-8:], 'decision_step_limit': limit,
                        'capabilities': {'run_groot': True, 'set_gripper': tools.allow_gripper},
                    })

                reason = 'plan_only' if args.plan_only else run_subtasks(
                    tools, plan, args.mode, decide, record,
                    args.decision_steps, args.max_decisions, args.max_replans,
                )
                result = {
                    'episode': episode, 'seed': args.seed + episode, 'task': args.task, 'mode': args.mode,
                    'success': tools.success, 'stop_reason': reason, 'steps': tools.steps,
                    'api_calls': api_calls, 'wall_seconds': time.monotonic() - started,
                    'inference_seconds': tools.inference_seconds, 'original_goal': goal,
                }
                (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                results.append(result)
        except Exception as error:
            (directory / 'error.json').write_text(json.dumps({'error': str(error)}, indent=2) + '\n')
            raise
        finally:
            if env is not None:
                env.close()
    summary = {
        'task': args.task, 'mode': args.mode, 'episodes': len(results),
        'successes': sum(item['success'] for item in results),
        'success_rate': sum(item['success'] for item in results) / len(results),
        'evaluation': 'Custom diagnostic; success is the environment task check, not Muse verification.',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
