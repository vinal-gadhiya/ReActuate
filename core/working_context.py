"""One compact public model view; complete audit context stays on disk."""

from copy import deepcopy
import re


def words(value):
    return set(re.findall(r'[a-z]+', str(value).casefold())) - {'the', 'a', 'on', 'in', 'to', 'and', 'of'}


def task_references(catalog, context, limit=10):
    scene = context.get('simulator_context') or {}
    task = scene.get('task') or {}
    reference = task.get('composite_reference') or {}
    stage = context.get('step') or {}
    query = words([context.get('original_goal', ''), stage, reference.get('skills_required', [])])
    ranked = sorted(catalog['tasks'], key=lambda entry: (
        -len(query & words([entry['name'], entry['category'], entry['description']])), entry['name']))
    return {'upstream_ref': catalog['upstream_ref'], 'source': catalog['source'],
            'tasks': deepcopy(ranked[:limit]),
            'interpretation': 'Relevant reference wording only. Omission does not disable tools or imply lack of policy capability.'}


def working_context(context):
    scene = context.get('simulator_context') or {}
    task = scene.get('task') or {}
    active = context.get('step')
    ledger = context.get('goal_ledger')
    query = words([context.get('original_goal', ''), active, ledger])
    entries = (context.get('scene_memory') or {}).get('entries', [])
    entries = sorted(entries, key=lambda item: (
        -len(query & words([item.get('name'), item.get('location')])), -item.get('observation_step', 0)))[:16]
    beliefs = [{key: deepcopy(item.get(key)) for key in
                ('name', 'kind', 'location', 'status', 'evidence', 'observation_step', 'previous_location_claim')}
               for item in entries]
    for item in beliefs:
        item['age_steps'] = max(0, context.get('observation_step', 0) - (item['observation_step'] or 0))
    selected = {key: deepcopy(context[key]) for key in (
        'original_goal', 'goal_ledger', 'step', 'next_step', 'subtask_steps_used',
        'subtask_budget_mode', 'planning_window_exhausted', 'candidate_plan', 'diagnostic_replay',
        'observation_step', 'before_observation_step', 'observation_refs', 'depth_observations',
        'attempt_history', 'recovery_feedback', 'plan_validation_feedback', 'allowed_current_cameras',
        'decision_step_limit', 'episode_steps_left', 'api_calls_left', 'replans_left',
        'navigation_bursts_left', 'navigation_budget_mode', 'execution_limits', 'capabilities',
        'control_owner', 'robot_measurements', 'supervisor_stop_policy',
    ) if key in context}
    selected.update({
        'original_goal': context.get('original_goal') or (context.get('episode_context') or {}).get('original_goal', ''),
        'version': 2, 'beliefs': beliefs,
        'simulator_context': {'task': deepcopy(task), 'robot': deepcopy(scene.get('guide', {}).get('robot', {})),
                              'tools': deepcopy(scene.get('tools', {})), 'cameras': deepcopy(scene.get('cameras', [])),
                              'operator_context': deepcopy(scene.get('operator_context', {})), 'policy_reference': {}},
        'belief_rule': 'Prefer current observations over stale claims. Conflicts remain uncertain; age is simulated control steps, not wall time.',
    })
    return selected
