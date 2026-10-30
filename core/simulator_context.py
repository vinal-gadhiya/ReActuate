"""Public simulator/capability context; never inspect task-private object state."""

import json
from copy import deepcopy
from pathlib import Path

CONTEXT_VERSION = 6


def load_atomic_task_catalog(registered_tasks=()):
    """Pinned public description templates, with coverage checked against the runtime registry."""
    catalog = json.loads(Path(__file__).with_name('atomic_task_catalog.json').read_text())
    tasks = catalog['tasks']
    names = [task['name'] for task in tasks]
    if len(names) != len(set(names)) or any(not task['description'].strip() for task in tasks):
        raise ValueError('Atomic instruction catalog contains duplicate or empty entries')
    missing = sorted(set(registered_tasks) - set(names))
    if missing:
        raise ValueError(f'Atomic instruction catalog does not cover registered tasks: {missing}')
    return catalog


def load_composite_task_catalog(registered_tasks=()):
    """Pinned descriptions and the official Skills Involved display labels."""
    catalog = json.loads(Path(__file__).with_name('composite_task_catalog.json').read_text())
    names = set()
    for task in catalog['tasks']:
        name = task['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Composite task catalog contains duplicate or empty names')
        names.add(name)
        if not isinstance(task['description'], str) or not task['description'].strip():
            raise ValueError(f'Composite task {name} has no description')
        for field in ('skills_required', 'skill_keys'):
            skills = task[field]
            if (not isinstance(skills, list)
                    or any(not isinstance(skill, str) or not skill.strip() for skill in skills)
                    or len(skills) != len(set(skills))):
                raise ValueError(f'Composite task {name} has invalid {field}')
        if len(task['skills_required']) != len(task['skill_keys']):
            raise ValueError(f'Composite task {name} has mismatched skill labels and keys')
    missing = sorted(set(registered_tasks) - names)
    if missing:
        raise ValueError(f'Composite task catalog does not cover registered tasks: {missing}')
    return catalog


def composite_task_reference(task, registered_tasks=()):
    catalog = load_composite_task_catalog(registered_tasks)
    entry = next((entry for entry in catalog['tasks'] if entry['name'] == task), None)
    if entry is None:
        raise ValueError(f'Composite task catalog does not cover current task: {task}')
    return {
        **deepcopy(entry),
        **{key: catalog[key] for key in ('upstream_ref', 'description_source', 'skills_source',
                                         'navigation_source', 'description_kind', 'skills_kind')},
        'usage': 'Use these unordered skill hints with the atomic instruction catalog to check action '
                 'coverage before planning and when replanning. Ground actions and order in the current '
                 'scene and original_goal. A label does not require an unnecessary action; absence of a '
                 'label does not forbid a needed action. These are not tools, a fixed recipe, success '
                 'predicates, or proof of policy competence.',
    }


SIMULATOR_GUIDE = {
    'simulator': 'RoboCasa kitchen manipulation on MuJoCo',
    'robot': {
        'embodiment': 'PandaOmron',
        'description': 'A wheeled mobile base with one Panda arm and a parallel-jaw gripper; it does not walk.',
        'executor': 'Frozen GR00T N1.5 predicts arm, gripper, and base actions from current images, proprioception, and language.',
    },
    'observation_rules': [
        'RGB images are partial views, not a complete scene inventory.',
        'Proprioception describes the robot, not ground-truth object locations or contacts.',
        'No hidden object poses, contact labels, or task-success predicates are supplied. '
        'Optional depth views measure visible surfaces only; their scale is described in depth_observations.',
        'A task-mentioned object may be occluded or inside a container; not visible does not mean absent.',
        'A visible object may be a distractor. Match task attributes and relations before selecting it.',
        'SAM candidates are perception estimates, not proof of identity, contact, or containment.',
    ],
    'execution_rules': [
        'Subtasks execute in the original task environment. Atomic-task names do not create or switch environments.',
        'Physical scene state persists across subtask transitions and replans.',
        'Independent unfinished goals may be reordered to use a visible or already-held required object. Preserve true dependencies, completed work, and every remaining goal.',
        'Only listed tools exist. navigate wraps a fallible GR00T language request; no navigation planner, move_eef, IK solver, or manual pick/place primitive is installed. Separate pickup and placement language requests use run_groot.',
        'A run_groot instruction is a request to a fallible policy, not a guaranteed skill.',
        'Named fixture navigation does not require visual localization, a memory entry, or an original-task mention. The policy resolves the route; a request is not visual confirmation or verified arrival.',
        'Do not invent tools or interactions. Stop or replan if prerequisites cannot be observed or executed.',
        'Avoid task-irrelevant cleanup. Release placed objects and move clear when needed to establish completion.',
        'Use observable stopping conditions: lift-and-move evidence is stronger than gripper closure alone.',
        'Subtask max_steps are ceilings in plan budget mode and advisory in episode mode; episode, decision, and API limits still apply.',
        'Navigation has no separate burst cap by default and uses remaining execution budgets. '
        'An explicit cap still applies; repeated stationary requests are rejected. Simple Navigate-to '
        'language is routed through the same navigation tool and checks.',
    ],
}


def load_operator_context(path):
    if path is None:
        return {}
    if path.stat().st_size > 16000:
        raise ValueError('Context file must be at most 16 KB')
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) - {'task_notes', 'scene_notes', 'constraints'}:
        raise ValueError('Context file accepts task_notes, scene_notes, and constraints only')
    for key in ('task_notes', 'scene_notes'):
        if key in value and (not isinstance(value[key], str) or len(value[key]) > 4000):
            raise ValueError(f'{key} must be a string of at most 4000 characters')
    if 'constraints' in value:
        constraints = value['constraints']
        if not isinstance(constraints, list) or len(constraints) > 20 or any(
            not isinstance(item, str) or not 0 < len(item.strip()) <= 500 for item in constraints
        ):
            raise ValueError('constraints must contain at most 20 short strings')
    return value


def build_simulator_context(task, is_composite, metadata, atomic_tasks, allow_gripper, operator_context,
                            composite_tasks=()):
    atomic_tasks = sorted(atomic_tasks)
    catalog = load_atomic_task_catalog(atomic_tasks)
    guide = SIMULATOR_GUIDE
    if metadata.get('policy') == 'patchwam':
        guide = deepcopy(SIMULATOR_GUIDE)
        guide['robot']['executor'] = 'Frozen PatchWAM predicts arm, gripper, and base actions from current images, proprioception, and language.'
        guide['execution_rules'] = [rule.replace('GR00T', 'PatchWAM') for rule in guide['execution_rules']]
    return {
        'version': CONTEXT_VERSION, 'guide': guide,
        'task': {'name': task, 'kind': 'composite' if is_composite else 'atomic',
                 'composite_reference': composite_task_reference(task, composite_tasks) if is_composite else None,
                 'instruction_source': 'original_goal is the episode description, not a generic task summary'},
        'policy_reference': {
            'registered_atomic_task_names': atomic_tasks,
            'meaning': ('Task-category vocabulary only, not tools, scene entities, or verified PatchWAM competencies.'
                        if metadata.get('policy') == 'patchwam' else
                        'Task-category vocabulary only, not tools, scene entities, or verified GR00T competencies.'),
            'instruction_catalog': {
                'upstream_ref': catalog['upstream_ref'],
                'source': catalog['description_source'],
                'description_kind': catalog['description_kind'],
                'tasks': [{key: entry[key] for key in ('name', 'category', 'description')}
                          for entry in catalog['tasks']],
                'usage': 'Use this vocabulary to form concise single-action instructions for the actual goal. '
                         'Separate pickup and placement; keep detailed grounding in memory and success criteria. Templates are not executable tools '
                         'or evidence of checkpoint competence. Exact demonstration annotations may differ.',
            },
        },
        'cameras': [{
            'key': key, 'role': 'wrist camera' if 'eye_in_hand' in key else 'external robot camera',
            'width': metadata['camera_width'], 'height': metadata['camera_height'],
        } for key in metadata['modalities']['video']['modality_keys']],
        'tools': {
            'run_groot': {'available': True, 'arguments': 'instruction: str, steps: positive int',
                          'meaning': 'A short policy burst in the current scene'},
            'navigate': {'available': True, 'arguments': 'fixture: str, steps: positive int',
                         'meaning': 'Bounded NavigateKitchen request to a named fixture, including unseen destinations; the policy resolves navigation'},
            'set_gripper': {'available': bool(allow_gripper), 'arguments': 'closed: bool, steps: int in [1,10]',
                            'meaning': 'Gripper command with zero commanded arm/base delta; not a grasp skill'},
        },
        'operator_context': {'source': 'operator supplied; not independently verified', 'notes': operator_context},
    }
