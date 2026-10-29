"""Bounded in-context planning journal from public decisions and execution events.

This summarizes explicit evidence/rationales, not private model reasoning. It
does not query the simulator, call a model, or generate robot actions.
"""

from collections import OrderedDict, deque
from copy import deepcopy
import json


def short(value, limit=300):
    value = str(value)
    return value if len(value) <= limit else value[:limit - 1] + '…'


def plan_view(steps):
    return [{'id': short(step['id'], 100), 'instruction': short(step['instruction'], 240),
             'target': short(step['target'], 80),
             'success_criteria': short(step['success_criteria'], 160),
             'max_steps': step['max_steps']} for step in steps]


class EpisodeContext:
    VERSION = 1
    MAX_CHARS = 24000

    def __init__(self):
        self.goal = ''
        self.initial_plan = []
        self.active_plan = []
        self.completed = OrderedDict()
        self.attempts = OrderedDict()
        self.facts = OrderedDict()
        self.changes = deque(maxlen=12)
        self.recent = deque(maxlen=12)
        self.revisions = deque(maxlen=4)
        self.recoveries = deque(maxlen=8)
        self.revision = 0
        self.total_steps = self.total_bursts = self.total_decisions = 0
        self.recovery_count = 0
        self.event_count = 0
        self.latest_decision = None

    def observe(self, event):
        kind = event['event']
        step, subtask = event.get('step', event.get('step_after', 0)), event.get('subtask', '')
        item = {'event': kind, 'step': step, 'subtask': subtask}
        if kind == 'episode_started':
            self.goal = short(event['goal'], 2000)
        elif kind == 'initial_plan':
            self.goal = short(event['goal'], 2000)
            self.initial_plan = deepcopy(event['steps'])
            self.active_plan = deepcopy(event['steps'])
        elif kind == 'plan_replaced':
            # Only accepted runner events change the plan, never a proposal.
            self.active_plan = deepcopy(event['steps'])
            self.revision += 1
            item.update(reason=short(event['reason']), revision=self.revision)
            self.revisions.append(dict(item))
        elif kind == 'decision':
            self.total_decisions += 1
            item.update(status=event['status'], evidence=short(event['evidence']))
            assessment = event.get('assessment', {})
            item.update(phase=assessment.get('phase'), held_object=assessment.get('held_object'),
                        grasp=assessment.get('grasp'))
            self.latest_decision = dict(item)
            for entry in event.get('memory_updates', []):
                key = (entry['kind'], entry['name'].casefold().strip())
                signature = (entry['location'], entry['status'])
                old = self.facts.get(key)
                if old is None or old['signature'] != signature:
                    change = {'step': step, 'name': short(entry['name'], 100), 'kind': entry['kind'],
                              'location': short(entry['location'], 180), 'status': entry['status'],
                              'evidence': short(entry['evidence'], 220), 'source': 'Muse visual claim'}
                    if old is not None:
                        change['previous_location'] = short(old['location'], 180)
                        change['previous_status'] = old['status']
                    self.changes.append(change)
                self.facts[key] = {'signature': signature, 'location': entry['location'], 'status': entry['status']}
                self.facts.move_to_end(key)
                if len(self.facts) > 48:
                    self.facts.popitem(last=False)
        elif kind == 'tool':
            count = event['step_after'] - event['step_before']
            arguments = event['arguments']
            instruction = arguments.get('instruction', f'Navigate to the {arguments.get("fixture", "")}.')
            item.update(tool=event['tool'], instruction=short(instruction, 240),
                        step_before=event['step_before'], executed_steps=count)
            attempt = self.attempts.setdefault(subtask, {'subtask': subtask, 'bursts': 0, 'executed_steps': 0})
            attempt.update(bursts=attempt['bursts'] + 1, executed_steps=attempt['executed_steps'] + count,
                           last_instruction=short(instruction, 240), last_execution_step=step)
            self.attempts.move_to_end(subtask)
            if len(self.attempts) > 32:
                self.attempts.popitem(last=False)
            self.total_steps += count
            self.total_bursts += 1
        elif kind == 'subtask_transition':
            accepted = {'id': event['completed_step']['id'],
                        'instruction': short(event['completed_step']['instruction'], 180),
                        'step': step, 'evidence': short(event['completion_evidence']),
                        'source': 'Muse completion claim accepted by runner; not environment proof'}
            self.completed[accepted['id']] = accepted
            item.update(evidence=accepted['evidence'])
        elif kind in ('invalid_decision', 'recovery_required', 'transition_blocked',
                      'supervisor_unavailable', 'supervisor_stop_deferred', 'supervisor_failure_continuation'):
            self.recovery_count += 1
            item['reason'] = short(event.get('reason', event.get('error', event.get('evidence', ''))))
            self.recoveries.append(dict(item))
        else:
            return False
        self.recent.append(item)
        self.event_count += 1
        return True

    def context(self, observation_step=0):
        completed_ids = set(self.completed)
        value = {
            'version': self.VERSION, 'observation_step': observation_step,
            'original_goal': self.goal, 'plan_revision': self.revision,
            'initial_plan': plan_view(self.initial_plan),
            'unfinished_plan': plan_view([s for s in self.active_plan if s['id'] not in completed_ids]),
            'completed_claims': list(self.completed.values()),
            'plan_changes': list(self.revisions),
            'recovery_feedback': list(self.recoveries),
            'goal_attempts': list(self.attempts.values()),
            'latest_decision': self.latest_decision,
            'object_location_changes': list(self.changes),
            'recent_events': list(self.recent),
            'totals': {'executed_steps': self.total_steps, 'policy_bursts': self.total_bursts,
                       'decisions': self.total_decisions, 'recovery_events': self.recovery_count,
                       'accepted_completions': len(self.completed), 'journal_events': self.event_count},
            'interpretation': 'Read with current images, proprioception, scene_memory.entries and active step. '
                              'This is a bounded summary of explicit model outputs and executed events, not hidden reasoning. '
                              'Plans are intentions; executions are attempts; visual memories/completions are fallible claims. '
                              'Older locations may be stale. Full event history is saved in trace.jsonl. '
                              'No cross-episode memory or simulator-private facts are used.',
            'omitted_summary_records': 0,
        }
        value = deepcopy(value)
        # Prefer retaining the current unfinished plan and latest decision. Full
        # older records remain available on disk; never add extra summary API calls.
        for key in ('recent_events', 'object_location_changes', 'goal_attempts', 'initial_plan',
                    'completed_claims', 'recovery_feedback', 'plan_changes'):
            while value[key] and len(json.dumps(value, ensure_ascii=False)) > self.MAX_CHARS:
                value[key].pop(0)
                value['omitted_summary_records'] += 1
        return value
