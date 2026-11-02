"""One compact public model view; complete audit context stays on disk."""

from copy import deepcopy
import re
import json
from core.selective_review import previous_transport


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
    # Reserve references for the task's documented skills before filling lexical
    # matches. This is catalog retrieval, not a task-specific execution recipe.
    forms = catalog.get('instruction_forms', [])
    selected = [entry for entry in ranked if entry['name'] == task.get('name')]
    for skill in reference.get('skill_keys', []):
        sources = set(catalog.get('skill_reference_tasks', {}).get(skill, []))
        sources.update(name for form in forms if skill in form['skill_keys'] for name in form['source_tasks'])
        if any(entry['name'] in sources for entry in selected):
            continue
        match = next((entry for entry in ranked if entry['name'] in sources), None)
        if match is not None:
            selected.append(match)
    selected.extend(entry for entry in ranked if entry not in selected)
    return {'upstream_ref': catalog['upstream_ref'], 'source': catalog['source'],
            'tasks': deepcopy(selected[:limit]),
            'interpretation': 'Atomic descriptions supply required instruction vocabulary and format. '
                              'Composite skill hints guide retrieval, not compulsory order or proof of policy competence.'}


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
    # Keep a bounded identity trail. A hidden wrist view must not erase a
    # cross-camera retention claim, nor should an old claim become fresh truth.
    identity_chars_left = 4000
    for belief, source in zip(beliefs[:6], entries[:6]):
        trail = {'identity_anchor': deepcopy(source.get('identity_anchor')),
                 'observations': deepcopy(source.get('observations', [])[-2:])}
        size = len(json.dumps(trail))
        if size <= identity_chars_left:
            belief.update(trail)
            identity_chars_left -= size
    selected = {key: deepcopy(context[key]) for key in (
        'original_goal', 'goal_ledger', 'step', 'next_step', 'subtask_steps_used', 'last_stage_transition', 'last_stage_dispute',
        'subtask_budget_mode', 'planning_window_exhausted', 'candidate_plan', 'diagnostic_replay',
        'observation_step', 'before_observation_step', 'observation_refs', 'depth_observations',
        'temporal_observations', 'boundary_review_enabled', 'verification_mode', 'verification_feedback',
        'attempt_history', 'recovery_feedback', 'plan_validation_feedback', 'allowed_current_cameras',
        'decision_step_limit', 'episode_steps_left', 'api_calls_left', 'replans_left',
        'navigation_bursts_left', 'navigation_budget_mode', 'execution_limits', 'capabilities',
        'control_owner', 'robot_measurements', 'supervisor_stop_policy', 'execution_policy',
    ) if key in context}
    selected.update({
        'original_goal': context.get('original_goal') or (context.get('episode_context') or {}).get('original_goal', ''),
        'version': 2, 'beliefs': beliefs,
        'previous_transport_claim': previous_transport(context),
        'simulator_context': {'task': deepcopy(task), 'robot': deepcopy(scene.get('guide', {}).get('robot', {})),
                              'tools': deepcopy(scene.get('tools', {})), 'cameras': deepcopy(scene.get('cameras', [])),
                              'operator_context': deepcopy(scene.get('operator_context', {})), 'policy_reference': {}},
        'belief_rule': 'Prefer current observations over stale claims. Conflicts remain uncertain; age is simulated control steps, not wall time.',
    })
    feedback = context.get('execution_feedback') or {}
    selected['recent_execution_feedback'] = {
        'instruction_attempts': deepcopy(feedback.get('instruction_attempts', [])[-3:]),
        'recent_outcome_reviews': deepcopy(feedback.get('recent_outcome_reviews', [])[-2:]),
        'interpretation': feedback.get('interpretation', ''),
    }
    return selected
