"""Persistent public requirements and observed attempts, independent of plan IDs.

All outcomes are dated model claims. This module never queries simulator state,
rewrites policy language, or chooses robot actions.
"""

from copy import deepcopy


def record(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


REF_SCHEMA = record({'step': {'type': 'integer', 'minimum': 0}, 'camera': {'type': 'string'}})
REFS_SCHEMA = {'type': 'array', 'items': REF_SCHEMA}
REQUIREMENT_SCHEMA = record({'id': {'type': 'string'}, 'description': {'type': 'string'},
                             'dependencies': {'type': 'array', 'items': {'type': 'string'}}})
GOAL_UPDATE_SCHEMA = record({'goal_id': {'type': 'string'},
                             'status': {'type': 'string', 'enum': ['satisfied', 'unsatisfied', 'unknown']},
                             'evidence': {'type': 'string'}, 'observation_refs': REFS_SCHEMA})
OUTCOMES = ('intended_object_acquired', 'other_required_object_acquired', 'unrelated_object_acquired', 'prerequisite_changed',
            'progress', 'no_reported_progress', 'uncertain')
ATTEMPT_REPORT_SCHEMA = record({'attempt_id': {'type': 'string'},
                                'outcome': {'type': 'string', 'enum': list(OUTCOMES)},
                                'evidence': {'type': 'string'}, 'observation_refs': REFS_SCHEMA})
RECOVERY_SCHEMA = record({'change_type': {'type': 'string', 'enum': ['strategy', 'prerequisite']},
                          'description': {'type': 'string'}, 'observation_refs': REFS_SCHEMA,
                          'necessity': {'type': 'string', 'enum': [
                              'blocked_access', 'wrong_target', 'object_loss', 'unsafe_state',
                              'goal_contradiction', 'useful_required_object_held', 'not_needed']}})
LEGACY_RECOVERY_FIELDS = {'change_type', 'description', 'observation_refs'}
ACTION_SCHEMA = record({'intent': {'type': 'string'}, 'object': {'type': 'string'},
                        'destination': {'type': 'string'}, 'instruction': {'type': 'string'}})


def normalized(text):
    """History key only. The executed instruction is never changed."""
    return ' '.join(text.casefold().split()).rstrip('.!?')


def camera_key(camera):
    for prefix in ('current:', 'before:'):
        if camera.startswith(prefix):
            return camera[len(prefix):]
    return camera


def validate_requirements(requirements):
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 128:
        raise ValueError('Extract 1–128 explicit original requirements, independently of the short action window')
    by_id = {}
    for item in requirements:
        if not isinstance(item, dict) or set(item) != {'id', 'description', 'dependencies'}:
            raise ValueError('Requirements need stable id, description and dependencies')
        for key in ('id', 'description'):
            if not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > 2000:
                raise ValueError(f'Invalid requirement {key}')
        if item['id'] in by_id or any(normalized(old['description']) == normalized(item['description']) for old in by_id.values()):
            raise ValueError('Duplicate requirement identity or description')
        deps = item['dependencies']
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps) or len(deps) != len(set(deps)):
            raise ValueError('Invalid requirement dependencies')
        by_id[item['id']] = item
    visited, visiting = set(), set()

    def visit(identifier):
        if identifier not in by_id:
            raise ValueError(f'Unknown dependency: {identifier}')
        if identifier in visiting:
            raise ValueError('Requirement dependencies must be acyclic')
        if identifier in visited:
            return
        visiting.add(identifier)
        for dep in by_id[identifier]['dependencies']:
            visit(dep)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in by_id:
        visit(identifier)
    return deepcopy(requirements)


class GoalLedger:
    def __init__(self, original_goal, requirements):
        self.original_goal = original_goal
        self.requirements = {item['id']: {**item, 'status': 'unknown', 'evidence': '', 'observation_refs': []}
                             for item in validate_requirements(requirements)}
        self.observations = {}
        self.history = []

    def observe(self, step, cameras):
        self.observations.setdefault(step, set()).update(camera_key(camera) for camera in cameras)

    def validate_refs(self, refs, current_step, minimum_step=0):
        if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
            raise ValueError('Outcome claims require 1–12 actual observation references')
        for ref in refs:
            if (not isinstance(ref, dict) or set(ref) != {'step', 'camera'}
                    or type(ref['step']) is not int or not minimum_step <= ref['step'] <= current_step
                    or not isinstance(ref['camera'], str)
                    or camera_key(ref['camera']) not in self.observations.get(ref['step'], set())):
                raise ValueError('Outcome reference must identify a supplied observation at an eligible step')

    def apply_updates(self, updates, current_step):
        if not isinstance(updates, list) or len(updates) > 128:
            raise ValueError('Invalid requirement update list')
        candidate = deepcopy(self.requirements)
        touched = set()
        for update in updates:
            if not isinstance(update, dict) or set(update) != set(GOAL_UPDATE_SCHEMA['properties']):
                raise ValueError('Requirement updates may change status/evidence, never definitions or IDs')
            identifier = update['goal_id']
            if identifier not in candidate or identifier in touched:
                raise ValueError('Unknown or duplicate goal update ID')
            if update['status'] not in ('satisfied', 'unsatisfied', 'unknown') or not isinstance(update['evidence'], str) or not update['evidence'].strip():
                raise ValueError('Requirement outcome needs a status and evidence')
            previous = candidate[identifier]
            last_step = max((ref['step'] for ref in previous['observation_refs']), default=-1)
            self.validate_refs(update['observation_refs'], current_step)
            if max(ref['step'] for ref in update['observation_refs']) < last_step:
                raise ValueError('Requirement update needs evidence at least as recent as its last assessment')
            if update['status'] == 'satisfied' and max(ref['step'] for ref in update['observation_refs']) != current_step:
                raise ValueError('Requirement satisfaction needs current observation evidence')
            if (previous['status'] == 'satisfied' and update['status'] != 'satisfied'
                    and max(ref['step'] for ref in update['observation_refs']) <= last_step):
                raise ValueError('Reopening a satisfied requirement needs newer contradictory observation evidence')
            candidate[identifier].update({key: deepcopy(update[key]) for key in ('status', 'evidence', 'observation_refs')})
            touched.add(identifier)
        for identifier in touched:
            if candidate[identifier]['status'] == 'satisfied' and any(
                    candidate[dep]['status'] != 'satisfied' for dep in candidate[identifier]['dependencies']):
                raise ValueError('A satisfied requirement has unmet declared dependencies')
        self.requirements = candidate
        self.history.extend({'step': current_step, **deepcopy(update)} for update in updates)

    def validate_stage(self, stage):
        ids = stage.get('goal_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
            raise ValueError('Every execution stage must reference stable goal_ids')
        if any(identifier not in self.requirements for identifier in ids):
            raise ValueError('Stage references a goal absent from the persistent ledger')
        if all(self.requirements[identifier]['status'] == 'satisfied' for identifier in ids):
            raise ValueError('Stage duplicates already satisfied work; reopen the requirement with new evidence first')

    def validate_window(self, stages):
        for stage in stages:
            self.validate_stage(stage)

    def pending(self):
        return [identifier for identifier, item in self.requirements.items() if item['status'] != 'satisfied']

    def definitions(self):
        return [{key: deepcopy(item[key]) for key in ('id', 'description', 'dependencies')} for item in self.requirements.values()]

    def context(self):
        return {'original_goal': self.original_goal, 'requirements': deepcopy(list(self.requirements.values())),
                'pending_goal_ids': self.pending(), 'interpretation': 'Persistent original requirements. Status and evidence are model claims, not simulator truth. Replanning cannot edit definitions.'}


class AttemptHistory:
    def __init__(self, ledger, failure_limit=3, recovery_limit=2):
        if failure_limit < 1 or recovery_limit < 0:
            raise ValueError('Attempt limits require positive failures and nonnegative recoveries')
        self.ledger, self.failure_limit, self.recovery_limit = ledger, failure_limit, recovery_limit
        self.attempts = []
        self.recoveries = []

    @staticmethod
    def command_key(name, arguments):
        # Burst length and stage IDs do not define a new strategy.
        text = arguments.get('instruction', f'Navigate to the {arguments.get("fixture", "")}.')
        return normalized(text)

    def record(self, goal_ids, name, arguments, action, before, after, recovery=None):
        if after <= before:
            return None
        attempt = {'attempt_id': f'attempt-{len(self.attempts) + 1:05d}', 'goal_ids': list(goal_ids),
                   'command_key': self.command_key(name, arguments), 'tool': name,
                   'instruction': arguments.get('instruction', f'Navigate to the {arguments.get("fixture", "")}.'),
                   'semantic_action': deepcopy(action), 'step_before': before, 'step_after': after,
                   'outcome': 'uncertain', 'evidence': '', 'observation_refs': []}
        self.attempts.append(attempt)
        if recovery is not None:
            self.recoveries.append({'attempt_id': attempt['attempt_id'], 'goal_ids': list(goal_ids), **deepcopy(recovery)})
        return deepcopy(attempt)

    def review(self, reports, current_step):
        if not isinstance(reports, list) or len(reports) > 12:
            raise ValueError('At most 12 recent attempt reports per decision')
        candidate = deepcopy(self.attempts)
        touched = set()
        for report in reports:
            if not isinstance(report, dict) or set(report) != set(ATTEMPT_REPORT_SCHEMA['properties']):
                raise ValueError('Invalid observed attempt report')
            identifier = report['attempt_id']
            attempt = next((item for item in candidate if item['attempt_id'] == identifier), None)
            if attempt is None or identifier in touched or report['outcome'] not in OUTCOMES:
                raise ValueError('Unknown/duplicate attempt or invalid outcome')
            if not isinstance(report['evidence'], str) or not report['evidence'].strip():
                raise ValueError('Attempt outcome needs observation evidence')
            self.ledger.validate_refs(report['observation_refs'], current_step, attempt['step_before'])
            if max(ref['step'] for ref in report['observation_refs']) < attempt['step_after']:
                raise ValueError('Attempt review needs at least one after-action observation; before views may support comparison')
            # Repeated API assessments of one attempt never create new failures.
            if attempt['outcome'] != 'uncertain':
                if report['outcome'] != attempt['outcome']:
                    raise ValueError('A supported attempt report cannot be overwritten to reset failure history')
            # Equivalent outcomes may be paraphrased or gain new evidence without
            # inventing another failure or erasing the earlier dated claim.
            if any(report[key] != attempt[key] for key in ('outcome', 'evidence', 'observation_refs')):
                old = {key: deepcopy(attempt[key]) for key in ('outcome', 'evidence', 'observation_refs')}
                if old['observation_refs']:
                    attempt['previous_reviews'] = (attempt.get('previous_reviews', []) + [old])[-12:]
                attempt.update({key: deepcopy(report[key]) for key in ('outcome', 'evidence', 'observation_refs')})
            touched.add(identifier)
        self.attempts = candidate

    def failures(self, goal_id):
        result = []
        for attempt in self.attempts:
            if goal_id not in attempt['goal_ids']:
                continue
            if attempt['outcome'] in ('progress', 'intended_object_acquired'):
                result = []
            elif attempt['outcome'] in ('other_required_object_acquired', 'unrelated_object_acquired', 'no_reported_progress'):
                result.append(attempt)
        return result

    def authorize(self, goal_ids, name, arguments, recovery, current_step):
        blocked = [identifier for identifier in goal_ids if len(self.failures(identifier)) >= self.failure_limit]
        if not blocked:
            return None
        if not isinstance(recovery, dict) or set(recovery) not in (set(RECOVERY_SCHEMA['properties']), LEGACY_RECOVERY_FIELDS):
            raise ValueError('Repeated supported goal failures require a changed strategy/prerequisite with evidence, or stop')
        if recovery['change_type'] not in ('strategy', 'prerequisite') or not isinstance(recovery['description'], str) or not recovery['description'].strip():
            raise ValueError('Recovery must explain what changes')
        self.ledger.validate_refs(recovery['observation_refs'], current_step)
        authorization = deepcopy(recovery)
        progress_ids = []
        command = self.command_key(name, arguments)
        for identifier in blocked:
            if sum(identifier in item['goal_ids'] for item in self.recoveries) >= self.recovery_limit:
                raise ValueError('Goal recovery allowance exhausted; renaming stages does not reset it')
            failures = self.failures(identifier)
            if recovery['change_type'] == 'strategy':
                if any(item['command_key'] == command for item in failures):
                    raise ValueError('Recovery repeats a failed command; change actual execution or establish a changed prerequisite')
            else:
                last = failures[-1]['step_after']
                self.ledger.validate_refs(recovery['observation_refs'], current_step, last + 1)
                progress = [item for item in self.attempts if identifier in item['goal_ids']
                            and item['step_after'] > last and item['outcome'] == 'prerequisite_changed']
                if not progress:
                    raise ValueError('Prerequisite recovery needs a subsequent executed action with observed progress')
                if any(progress[-1]['attempt_id'] in item.get('progress_attempt_ids', []) for item in self.recoveries):
                    raise ValueError('The same prerequisite evidence cannot authorize repeated retries')
                progress_ids.append(progress[-1]['attempt_id'])
        authorization['progress_attempt_ids'] = sorted(set(progress_ids))
        return authorization

    def context(self, goal_ids=()):
        relevant = [item for item in self.attempts if not goal_ids or set(goal_ids) & set(item['goal_ids'])]
        return {'recent_attempts': deepcopy(relevant[-12:]),
                'failure_counts': {identifier: len(self.failures(identifier)) for identifier in self.ledger.requirements},
                'failure_limit': self.failure_limit, 'recovery_limit': self.recovery_limit,
                'interpretation': 'Observed outcomes remain fallible. History survives replans; instruction normalization is only an accounting key.'}
