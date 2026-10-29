"""Run a composite task with baseline, fixed-plan, or Muse-supervised control."""

import argparse
from collections import deque
from importlib.metadata import version
import json
from pathlib import Path
import time
import uuid

import numpy as np

from core.agent import parse_program, validate_decision, validate_plan
from core.policy_http import PolicyClient, SupervisorClient
from core.simulator_context import build_simulator_context, load_operator_context
from core.scene_memory import SceneMemory
from core.observations import check_rgb
from core.transport_control import ACTION_SHAPES, TransportControl
from sim.evaluate import make_policy_observation, validate_actions
from sim.scene_checks import choose_layout_style, ensure_hash_seed, reset_scene, scene_fingerprint, seed_process, verify_scene


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
    def __init__(self, env, observation, policy, metadata, video, max_steps, chunk_steps, allow_gripper=False,
                 artifact_directory=None, hybrid_control=False):
        self.env, self.observation, self.policy, self.metadata, self.video = env, observation, policy, metadata, video
        self.max_steps, self.chunk_steps = max_steps, chunk_steps
        self.steps, self.success, self.ended = 0, False, False
        self.inference_seconds = 0.0
        self.run_id = uuid.uuid4().hex
        self.allow_gripper = (allow_gripper or hybrid_control) and gripper_supported(env)
        if hybrid_control and not self.allow_gripper:
            raise ValueError('Hybrid control requires the validated fixed-impedance delta OSC controller')
        self.transport_control = TransportControl() if hybrid_control else None
        self.allow_vertical = hybrid_control
        if hybrid_control:
            self._validate_primitive_schema()
            controller = env.unwrapped.env.robots[0].composite_controller.part_controllers['right']
            if getattr(controller, 'input_ref_frame', None) not in ('base', 'world'):
                raise ValueError('Unknown controller translation frame')
            limits = np.asarray(controller.output_max)[:3]
            inputs = np.asarray(controller.input_max)[:3]
            if (limits.shape != (3,) or inputs.shape != (3,) or not np.all(np.isfinite(limits))
                    or not np.all(limits > 0) or not np.all(np.isfinite(inputs)) or not np.all(inputs > 0)):
                raise ValueError('Unknown controller translation scaling')
            self.translation_scale = limits / inputs
        self.camera = metadata['modalities']['video']['modality_keys'][0]
        self.frames = deque(maxlen=8)
        self.artifact_directory = artifact_directory
        self.last_diagnostics = {}
        self.frames.append((0, {key: observation[key].copy() for key in metadata['modalities']['video']['modality_keys']}))

    @property
    def done(self):
        return self.success or self.ended or self.steps >= self.max_steps

    def step(self, action, authorized_release=False):
        if self.done:
            return
        if self.transport_control is not None:
            action = self.transport_control.filter(action, authorized_release)
            if self.transport_control.phase == 'place':
                # Limit policy and primitive translation to 2 mm per control step.
                bound = 0.002 / self.translation_scale
                original = action['action.end_effector_position'].copy()
                action['action.end_effector_position'] = np.clip(original, -bound, bound)
                if np.any(original != action['action.end_effector_position']):
                    counts = self.transport_control.overrides
                    counts['placement_translation_limited'] = counts.get('placement_translation_limited', 0) + 1
        self.observation, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.success = bool(info.get('success', False))
        self.ended = bool(terminated or truncated)
        self.video.append_data(self.observation[self.camera])
        if self.steps % 8 == 0:
            self.frames.append((self.steps, {key: self.observation[key].copy()
                                            for key in self.metadata['modalities']['video']['modality_keys']}))
        return action

    def run_groot(self, instruction, steps):
        before = {key: np.asarray(self.observation[key]).copy() for key in
                  self.metadata['modalities']['state']['modality_keys']}
        image_before = self.observation[self.camera].copy()
        applied = []
        limit = min(self.steps + steps, self.max_steps)
        while self.steps < limit and not self.done:
            check_rgb({key: self.observation[key] for key in self.metadata['modalities']['video']['modality_keys']},
                      self.artifact_directory, self.steps)
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
                action = {key: value[index].copy() for key, value in actions.items()}
                applied.append(self.step(action))
                if self.done:
                    break
        # No queued actions survive this call or an instruction transition.
        changes = {key: float(np.max(np.abs(np.asarray(self.observation[key]) - value)))
                   for key, value in before.items()}
        image_change = float(np.mean(np.abs(self.observation[self.camera].astype(float) - image_before.astype(float))))
        self.last_diagnostics = {
            'state_max_abs_changes': changes, 'image_mean_abs_change': image_change,
            'action_mean_abs': {key: float(np.mean([np.abs(action[key]).mean() for action in applied]))
                                for key in applied[0]} if applied else {},
            'stationary': bool(changes) and max(changes.values()) < 1e-4 and image_change < 0.5,
            'note': 'Motion diagnostic only; motion is not evidence of task progress or grasp success.',
        }

    def navigate(self, fixture, steps):
        self.run_groot(f'Navigate to the {fixture}.', steps)

    def refresh_observation(self):
        wrapper = self.env.unwrapped
        raw = wrapper.env._get_observations(force_update=True)
        self.observation = wrapper.get_observation(raw)

    def perception_clip(self):
        frames = list(self.frames)
        current = {key: self.observation[key].copy() for key in self.metadata['modalities']['video']['modality_keys']}
        if frames[-1][0] == self.steps:
            frames[-1] = (self.steps, current)
        else:
            frames.append((self.steps, current))
        frames = frames[-8:]
        return [step for step, _ in frames], {f'clip:{key}': np.stack([view[key] for _, view in frames]) for key in current}

    def set_gripper(self, closed, steps):
        self.last_diagnostics = {'stationary': False, 'note': 'Direct gripper intervention; not a measured policy burst.'}
        if not self.allow_gripper:
            raise ValueError('Gripper primitive requires an explicitly enabled, validated delta OSC controller')
        self._validate_primitive_schema()
        if self.transport_control is not None:
            self.transport_control.authorize('set_gripper', {'closed': closed})
        for _ in range(steps):
            if self.done:
                break
            action = {key: np.zeros(shape, dtype=np.float32) for key, shape in ACTION_SHAPES.items()}
            action['action.gripper_close'][:] = float(closed)
            self.step(action, authorized_release=not closed)
        if not closed and self.transport_control is not None:
            self.transport_control.released()

    def _validate_primitive_schema(self):
        if set(self.env.action_space.spaces) != set(ACTION_SHAPES):
            raise ValueError('Unknown PandaOmron action schema')
        for key, shape in ACTION_SHAPES.items():
            if self.env.action_space.spaces[key].shape != shape:
                raise ValueError(f'Unknown action dimensions: {key}')

    def move_vertical(self, distance, steps):
        if not self.allow_vertical:
            raise ValueError('Vertical placement requires hybrid control')
        if not np.isfinite(distance) or not 0 < abs(distance) <= 0.02 or not 1 <= steps <= 10 or abs(distance) / steps > 0.002 + 1e-9:
            raise ValueError('Vertical motion is limited to 2 mm per step and 20 mm per call')
        self.transport_control.authorize('move_vertical', {})
        # PandaOmron has a planar mobile base: base/world z axes coincide.
        command = distance / steps / self.translation_scale[2]
        if abs(command) > self.env.action_space.spaces['action.end_effector_position'].high[2]:
            raise ValueError('Vertical request exceeds controller input range')
        self.last_diagnostics = {'stationary': False, 'note': 'Bounded vertical delta command; not measured displacement or contact.'}
        for _ in range(steps):
            if self.done:
                break
            action = {key: np.zeros(shape, dtype=np.float32) for key, shape in ACTION_SHAPES.items()}
            action['action.end_effector_position'][2] = command
            action['action.gripper_close'][:] = 1
            self.step(action)


def handoff_context(plan, index, history):
    """Keep accepted transition claims available after recent history rolls off."""
    return {
        'next_step': plan[index + 1] if index + 1 < len(plan) else None,
        'accepted_handoffs': [event for event in history if event['event'] == 'subtask_transition'][-10:],
        'handoff_rule': 'A prior handoff is a model claim at that time, not proof that access or grasp persists.',
    }


def run_subtasks(tools, plan, mode, decide, record, decision_steps=40, max_decisions=40, max_replans=2,
                 subtask_budget_mode='plan', scene_memory=None, max_navigation_bursts=6, stall_bursts=4,
                 placement_burst_steps=8):
    index, spent, decisions, replans, rejected = 0, 0, 0, 0, 0
    invalid_decisions = 0
    last_inspection, repeated_inspections = None, 0
    memory = scene_memory if scene_memory is not None else SceneMemory()
    while not tools.done and index < len(plan):
        step = plan[index]
        remaining = tools.max_steps - tools.steps
        if mode != 'agent' or subtask_budget_mode == 'plan':
            remaining = min(step['max_steps'] - spent, remaining)
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
            try:
                validate_decision(decision, max(0, min(decision_steps, remaining)), tools.allow_gripper,
                                  getattr(tools, 'allow_vertical', False))
            except (ValueError, KeyError, SyntaxError, TypeError) as error:
                invalid_decisions += 1
                record({'event': 'invalid_decision', 'step': tools.steps, 'reason': str(error),
                        'proposal': decision, 'consecutive_rejections': invalid_decisions,
                        'feedback': 'No motion or memory updates executed. Correct this proposal using available tools and current limits.'})
                if invalid_decisions >= 3:
                    return 'invalid_decision_limit'
                continue
            invalid_decisions = 0
            memory.update(decision, tools.steps, decisions)
            record({'event': 'scene_memory', 'step': tools.steps, 'memory': memory.context()})
        record({'event': 'decision', 'subtask': step['id'], 'step': tools.steps, **decision})
        transport = getattr(tools, 'transport_control', None)
        if transport is not None and mode == 'agent':
            transport.update(decision['assessment'], decision['memory_updates'])
            record({'event': 'transport_control', 'step': tools.steps, **transport.context()})
            if transport.loss_detected:
                return 'object_loss_reported'
        status = decision['status']
        if status == 'stop':
            return 'supervisor_stop'
        if status == 'inspect':
            inspection = (tools.steps, tuple(sorted(decision['perception'].items())))
            repeated_inspections = repeated_inspections + 1 if inspection == last_inspection else 1
            last_inspection = inspection
            if repeated_inspections >= 3:
                return 'perception_stall'
            if hasattr(tools, 'refresh_observation'):
                tools.refresh_observation()
            continue
        if status == 'complete':
            if mode == 'agent' and index + 1 < len(plan) and decision['handoff']['readiness'] != 'ready':
                record({'event': 'transition_blocked', 'subtask': step['id'],
                        'next_subtask': plan[index + 1]['id'], 'step': tools.steps,
                        'reason': 'Next-step prerequisites have not been visually established.',
                        'handoff': decision['handoff']})
                continue
            if mode == 'agent':
                record({'event': 'subtask_transition', 'subtask': step['id'],
                        'completed_step': dict(step), 'step': tools.steps,
                        'completion_evidence': decision['evidence'], 'handoff': decision['handoff']})
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
        commands = parse_program(decision['program'], min(decision_steps, remaining), tools.allow_gripper,
                                 getattr(tools, 'allow_vertical', False))
        try:
            if transport is not None:
                transport.authorize_program(commands)
            navigation_count = sum(name == 'navigate' for name, _ in commands)
            if mode == 'agent' and memory.navigation_bursts + navigation_count > max_navigation_bursts:
                raise ValueError('Navigation burst budget exhausted')
            for name, arguments in commands:
                if transport is not None:
                    transport.authorize(name, arguments)
                    if name == 'move_vertical' and abs(arguments['distance']) / arguments['steps'] > 0.002 + 1e-9:
                        raise ValueError('Use at least one control step per 2 mm of vertical motion')
                if name == 'navigate':
                    memory.authorize_navigation(arguments['fixture'], max_navigation_bursts)
                if (mode == 'agent' and name == 'run_groot'
                        and arguments['instruction'].strip().casefold().startswith('navigate ')):
                    raise ValueError('Use navigate(fixture=..., steps=...) for guarded navigation')
                if (mode == 'agent' and stall_bursts and memory.stall_streak >= stall_bursts
                        and name == 'run_groot' and arguments['instruction'] == memory.last_instruction):
                    raise ValueError('Stationary policy: choose information gathering, a distinct supported recovery, or stop')
        except ValueError as error:
            rejected += 1
            record({'event': 'recovery_required', 'step': tools.steps, 'reason': str(error)})
            if rejected >= 2:
                return 'recovery_unavailable'
            continue
        rejected = 0
        motion_left = placement_burst_steps if mode == 'agent' and memory.near_destination() else decision_steps
        if transport is not None and transport.held_object:
            motion_left = min(motion_left, 16)
        for name, arguments in commands:
            if motion_left <= 0:
                break
            if arguments['steps'] > motion_left:
                record({'event': 'bounded_execution', 'step': tools.steps,
                        'reason': 'Short feedback interval while holding an object or approaching placement',
                        'requested_steps': arguments['steps'], 'executed_step_limit': motion_left})
                if name == 'move_vertical':
                    arguments = {**arguments, 'distance': arguments['distance'] * motion_left / arguments['steps']}
                arguments = {**arguments, 'steps': motion_left}
            before = tools.steps
            getattr(tools, name)(**arguments)
            if mode == 'agent':
                diagnostics = getattr(tools, 'last_diagnostics', {})
                instruction = arguments.get('instruction', f'Navigate to the {arguments.get("fixture", "")}.')
                memory.note_motion(instruction, diagnostics)
                if name == 'navigate':
                    memory.navigation_bursts += 1
                    memory.last_navigation = arguments['fixture'].casefold().strip()
            spent += tools.steps - before
            motion_left -= tools.steps - before
            record({'event': 'tool', 'subtask': step['id'], 'tool': name, 'arguments': arguments,
                    'step_before': before, 'step_after': tools.steps,
                    'diagnostics': getattr(tools, 'last_diagnostics', {}),
                    **({'transport_control': transport.context()} if transport is not None else {})})
            if mode == 'agent':
                record({'event': 'scene_memory', 'step': tools.steps, 'memory': memory.context()})
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
    parser.add_argument('--context-file', type=Path, help='Optional public task/scene notes JSON')
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
    parser.add_argument('--max-navigation-bursts', type=int, default=6)
    parser.add_argument('--stall-bursts', type=int, default=4, help='Stationary bursts before requiring recovery; 0 disables')
    parser.add_argument('--placement-burst-steps', type=int, default=8)
    parser.add_argument('--subtask-budget-mode', choices=('plan', 'episode'), default='episode',
                        help='Agent only: enforce planned subtask ceilings, or use the remaining episode budget')
    parser.add_argument('--allow-gripper', action='store_true', help='Enable the guarded delta-OSC gripper primitive')
    parser.add_argument('--hybrid-control', action='store_true', help='Agent only: grasp latch, placement base stop, bounded vertical control and explicit release')
    parser.add_argument('--require-unseen-composite', action='store_true', help='Require task in pinned composite_unseen registry and outside pretrain300')
    parser.add_argument('--sam', action=argparse.BooleanOptionalAction, default=None,
                        help='Override supervisor SAM grounding for this run (--sam or --no-sam)')
    parser.add_argument('--sam-video', action='store_true', help='Opt into SAM clip tracking; default uses images only')
    parser.add_argument('--reference-scene', type=Path, help='Require matching baseline scene before executing actions')
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
    if args.max_navigation_bursts < 0 or args.stall_bursts < 0:
        parser.error('Navigation and stall limits must be nonnegative')
    if not 1 <= args.placement_burst_steps <= args.decision_steps:
        parser.error('Placement burst steps must be 1..decision-steps')
    if args.mode == 'fixed' and args.plan is None:
        parser.error('fixed mode requires --plan')
    if args.mode == 'baseline' and args.plan is not None:
        parser.error('baseline mode always uses the original task instruction')
    if args.hybrid_control and args.mode != 'agent':
        parser.error('--hybrid-control requires agent mode; baseline actions remain unmodified')
    if args.plan_only and args.mode != 'agent':
        parser.error('--plan-only requires agent mode')
    starting_plan = None
    operator_context = load_operator_context(args.context_file)
    if args.plan:
        starting_plan = validate_plan(json.loads(args.plan.read_text()))

    ensure_hash_seed(args.seed)
    seed_process(args.seed)
    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS, TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.models.scenes import scene_registry
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    if args.task not in ATOMIC_TASK_DATASETS and args.task not in COMPOSITE_TASK_DATASETS:
        parser.error('Unknown registered RoboCasa task')
    if args.require_unseen_composite and (args.task not in TASK_SET_REGISTRY['composite_unseen']
                                        or args.task in TASK_SET_REGISTRY['pretrain300']):
        parser.error('Task must be in composite_unseen and excluded from pretrain300')
    max_steps = args.max_steps or get_task_horizon(args.task)
    policy = PolicyClient(args.policy_url)
    metadata = policy.metadata()
    if metadata.get('data_config') != 'panda_omron' or metadata.get('embodiment') != 'PandaOmron':
        raise ValueError('A PandaOmron policy is required')
    if args.action_steps > metadata['action_horizon']:
        parser.error('Action steps exceed policy horizon')
    supervisor = SupervisorClient(args.supervisor_url, timeout=480) if args.mode == 'agent' else None
    supervisor_metadata = supervisor.metadata() if supervisor else None
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'resolved_max_steps': max_steps, 'policy': metadata, 'supervisor': supervisor_metadata,
                   'starting_plan': starting_plan, 'operator_context': operator_context,
                   'versions': {key: version(key) for key in ('robocasa', 'robosuite', 'mujoco', 'numpy')}})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    for episode in range(args.episodes):
        directory = args.output / f'episode-{episode:03d}'
        directory.mkdir()
        env = None
        tools = None
        api_calls = 0
        started = time.monotonic()
        history = []
        memory = SceneMemory()

        def record(event):
            if event['event'] == 'decision' and event.get('program'):
                filename = f'program-{len(history):03d}.py'
                (directory / filename).write_text(event['program'] + '\n')
                event = {**event, 'program_path': filename}
            with (directory / 'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            if event['event'] == 'scene_memory':
                (directory / 'scene_memory.json').write_text(json.dumps(event['memory'], indent=2) + '\n')
            if event['event'] in ('decision', 'tool', 'scheduled_transition', 'transition_blocked', 'subtask_transition', 'recovery_required', 'bounded_execution', 'invalid_decision'):
                history.append(event)
                print(json.dumps(event), flush=True)

        try:
            seed_process(args.seed + episode)
            task_class = REGISTERED_KITCHEN_ENVS[args.task]
            pairs = list(zip(range(1, 11), range(1, 11))) if args.split == 'target' else [
                (layout, style) for layout in scene_registry.unpack_layout_ids(-2)
                for style in scene_registry.unpack_style_ids(-2)]
            pair = choose_layout_style(pairs, task_class.EXCLUDE_LAYOUTS, task_class.EXCLUDE_STYLES, args.seed + episode)
            env = gym.make(
                f'robocasa/{args.task}', robots='PandaOmron', split=None, seed=args.seed + episode,
                obj_instance_split=args.split, layout_and_style_ids=[pair],
                camera_widths=metadata['camera_width'], camera_heights=metadata['camera_height'],
            )
            reference_scene = json.loads(args.reference_scene.read_text()) if args.reference_scene else None
            observation, _ = reset_scene(env, args.seed + episode, reference_scene)
            goal = str(observation['annotation.human.task_description'])
            camera_keys = metadata['modalities']['video']['modality_keys']
            scene = scene_fingerprint(env, observation, camera_keys)
            (directory / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
            if reference_scene is not None:
                record({'event': 'scene_verified', **verify_scene(scene, reference_scene)})
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
                clip_steps, clips = tools.perception_clip()
                # Clips are supplied only when SAM is enabled, not to the policy.
                if args.sam_video and (args.sam is True or (args.sam is None and supervisor_metadata['sam_enabled'])):
                    images.update(clips)
                quality = check_rgb(current, directory, tools.steps)
                references = []
                for number, (label, frame) in enumerate(sorted(images.items())):
                    if label.startswith('clip:'):
                        continue
                    filename = f'observation-{request_number:03d}-{number}.jpg'
                    imageio.imwrite(directory / filename, frame)
                    references.append({'label': label, 'path': filename})
                if any(key.startswith('clip:') for key in images):
                    clip_file = f'perception-{request_number:03d}.npz'
                    np.savez_compressed(directory / clip_file, **clips, steps=np.asarray(clip_steps))
                    references.append({'label': 'temporal camera samples', 'path': clip_file})
                decisions_used = sum(item['event'] in ('decision', 'invalid_decision') for item in history)
                decisions_left = max(0, args.max_decisions - decisions_used)
                calls_left = args.max_api_calls - api_calls
                motion_bound = min(max_steps - tools.steps, min(decisions_left, calls_left) * args.decision_steps)
                context.update({
                    'original_goal': goal, 'episode_steps_left': max_steps - tools.steps,
                    'api_calls_left': calls_left, 'simulator_context': simulator_context,
                    'scene_memory': memory.context(), 'perception_query': memory.perception,
                    'observation_step': tools.steps, 'request_number': request_number,
                    'clip_steps': clip_steps,
                    'navigation_bursts_left': max(0, args.max_navigation_bursts - memory.navigation_bursts),
                    'last_motion': tools.last_diagnostics,
                    'transport_control': tools.transport_control.context() if tools.transport_control is not None else None,
                    'image_diagnostics': quality, 'sam_video_enabled': args.sam_video,
                    'subtask_budget_mode': args.subtask_budget_mode,
                    'capabilities': {'run_groot': True, 'set_gripper': tools.allow_gripper,
                                     'move_vertical': tools.allow_vertical,
                                     'navigate': memory.navigation_bursts < args.max_navigation_bursts,
                                     'inspect': True,
                                     'sam': args.sam if args.sam is not None else supervisor_metadata['sam_enabled']},
                    'execution_limits': {
                        'step_unit': 'one executed environment control step, not seconds or API calls',
                        'control_frequency_hz': float(env.unwrapped.env.control_freq),
                        'decision_steps': args.decision_steps, 'decisions_left': decisions_left,
                        'api_calls_left': calls_left, 'motion_step_upper_bound': motion_bound,
                        'note': 'Optimistic bound: verification, planning, SAM, and replanning consume API calls too.',
                    },
                    'robot_measurements': {
                        key: np.asarray(tools.observation[key]).tolist()
                        for key in metadata['modalities']['state']['modality_keys']
                    },
                })
                if args.sam is not None:
                    context['use_sam'] = args.sam
                record({'event': 'supervisor_request', 'operation': operation, 'request': request_number,
                        'step': tools.steps, 'context': context, 'images': references})
                request_started = time.monotonic()
                response = supervisor.ask(operation, context, images)
                api_calls += response['api_calls']
                record({'event': 'supervisor_response', 'request': request_number,
                        'wall_seconds': time.monotonic() - request_started, **response})
                previous_images = current
                request_number += 1
                return response['result']

            with imageio.get_writer(directory / 'rollout.mp4', fps=int(env.unwrapped.env.control_freq)) as video:
                video.append_data(observation[camera_keys[0]])
                tools = EpisodeTools(env, observation, policy, metadata, video, max_steps, args.action_steps,
                                     args.allow_gripper, directory, args.hybrid_control)
                simulator_context = build_simulator_context(
                    args.task, args.task in COMPOSITE_TASK_DATASETS, metadata,
                    ATOMIC_TASK_DATASETS.keys(), tools.allow_gripper, operator_context,
                )
                simulator_context['tools']['move_vertical'] = {
                    'available': tools.allow_vertical,
                    'arguments': 'distance: signed metres in [-0.02,0.02], steps: 1..10; at most 0.002 metres per step',
                    'meaning': 'Small vertical delta in base/world z, gripper held closed, base stopped. Not IK, contact sensing, or guaranteed displacement.',
                }
                simulator_context['task']['benchmark_membership'] = (
                    'composite_unseen' if args.task in TASK_SET_REGISTRY['composite_unseen'] else 'other')
                simulator_context['task']['in_pretrain300'] = args.task in TASK_SET_REGISTRY['pretrain300']
                (directory / 'simulator_context.json').write_text(json.dumps(simulator_context, indent=2) + '\n')
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
                    if memory.near_destination():
                        limit = min(limit, args.placement_burst_steps)
                    if tools.transport_control is not None and tools.transport_control.held_object:
                        limit = min(limit, 16)
                    return ask('decide', {
                        'step': active_plan[index], 'subtask_steps_used': spent,
                        'completed_steps': active_plan[:index], 'remaining_steps': active_plan[index + 1:],
                        'history': history[-8:], 'decision_step_limit': limit,
                        **handoff_context(active_plan, index, history),
                        'capabilities': {'run_groot': True, 'set_gripper': tools.allow_gripper},
                    })

                reason = 'plan_only' if args.plan_only else run_subtasks(
                    tools, plan, args.mode, decide, record,
                    args.decision_steps, args.max_decisions, args.max_replans,
                    args.subtask_budget_mode,
                    memory, args.max_navigation_bursts, args.stall_bursts,
                    args.placement_burst_steps,
                )
                result = {
                    'episode': episode, 'seed': args.seed + episode, 'task': args.task, 'mode': args.mode,
                    'success': tools.success, 'stop_reason': reason, 'steps': tools.steps,
                    'api_calls': api_calls, 'wall_seconds': time.monotonic() - started,
                    'inference_seconds': tools.inference_seconds, 'original_goal': goal,
                    'scene_identity': scene['identity'],
                    'hybrid_control': args.hybrid_control,
                    'transport_control': tools.transport_control.context() if tools.transport_control is not None else None,
                    'benchmark_membership': simulator_context['task']['benchmark_membership'],
                }
                (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                results.append(result)
        except Exception as error:
            (directory / 'error.json').write_text(json.dumps({
                'error': str(error), 'step': tools.steps if tools is not None else 0,
                'completed_api_calls': api_calls,
                'note': 'A failed API request may have consumed usage not present in this count.',
                'scene_memory': memory.context(),
            }, indent=2) + '\n')
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
