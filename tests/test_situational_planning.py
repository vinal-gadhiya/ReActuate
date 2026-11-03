"""Task interpretation/progress persistence; no physical outcome verification."""

from copy import deepcopy
import json
import unittest

import numpy as np

from core.scene_memory import SceneMemory
from core.situational import run_situational, validate_situational
from core.situational_planning import update_planning
from inference.supervisor import Supervisor
from tests.test_situational import Tools, context, action


def initial():
    result = action()
    result['task_model'] = {'success_picture': 'Mugs on the right and bowls on the left of the named cabinet.',
        'requirements': [
            {'id': 'mugs', 'description': 'Gather the required mugs on the right.',
             'success_condition': 'All requested mugs rest released on the cabinet right side.',
             'reference_frame': 'Facing the named cabinet from its front.'},
            {'id': 'bowls', 'description': 'Gather the required bowls on the left.',
             'success_condition': 'All requested bowls rest released on the cabinet left side.',
             'reference_frame': 'Facing the named cabinet from its front.'}]}
    result['next_plan'] = ['Inspect neighboring compartments for remaining required items.',
                           'Group the mugs on the right and bowls on the left.']
    return result


class SituationalPlanningTests(unittest.TestCase):
    def test_initial_scene_survey_is_required_even_when_contents_are_unknown(self):
        with self.assertRaises(ValueError):
            update_planning({**initial(), 'scene_updates': []}, context())
        state = update_planning(initial(), context())
        self.assertEqual(state['scene_facts'][0]['basis'], 'unknown')
        self.assertTrue(state['open_questions'])

    def test_scene_facts_keep_dates_and_can_be_revised_from_new_observations(self):
        state = update_planning(initial(), context())
        result = action(step=32)
        state = update_planning(result, {**context(32), 'planning_state': state})
        self.assertEqual(state['scene_facts'][0]['observation_step'], 0)
        result = action(step=64)
        result['scene_updates'] = [{'id': 'access', 'description': 'Source compartment is now visible.',
            'basis': 'observed', 'observation_refs': context(64)['observation_refs']}]
        result['open_questions'] = []
        state = update_planning(result, {**context(64, 4000), 'planning_state': state})
        self.assertEqual(len(state['scene_facts']), 1)
        self.assertEqual(state['scene_facts'][0]['basis'], 'observed')
        self.assertEqual(state['scene_facts'][0]['observation_step'], 64)
        self.assertEqual(state['open_questions'], [])

    def test_new_scene_claim_cannot_use_old_or_invented_camera_evidence(self):
        state = update_planning(initial(), context())
        for refs in (context()['observation_refs'], [{'step': 32, 'camera': 'current:invented'}]):
            result = action(step=32)
            result['scene_updates'] = [{'id': 'access', 'description': 'Claimed source access.',
                                      'basis': 'observed', 'observation_refs': refs}]
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                update_planning(result, {**context(32), 'planning_state': state})

    def test_initial_model_and_plan_are_saved_before_any_motion(self):
        tools, events = Tools(horizon=32), []

        def ask(operation, payload):
            self.assertIsNone(payload['planning_state'])
            self.assertEqual(tools.steps, 0)
            payload.update(context(horizon=32))
            return initial()

        def record(event):
            if event['event'] == 'situational_planning':
                self.assertEqual(tools.steps, 0)
                self.assertEqual(len(event['state']['progress']), 2)
            events.append(event)

        run_situational(tools, ask, record, decision_steps=32, history=events, memory=SceneMemory())
        self.assertEqual([event['event'] for event in events],
                         ['situational_planning', 'situational_decision', 'tool'])

    def test_missing_initial_success_model_prevents_motion(self):
        tools, events = Tools(), []

        def ask(operation, payload):
            payload.update(context())
            return {**initial(), 'task_model': None}

        reason = run_situational(tools, ask, events.append, decision_steps=32,
                                 history=events, memory=SceneMemory(), max_failures=1)
        self.assertEqual(reason, 'supervisor_failure_limit')
        self.assertEqual(tools.steps, 0)

    def test_full_progress_survives_omitted_updates_and_changed_plans(self):
        state = update_planning(initial(), context())
        result = action(step=32)
        result['progress_updates'] = [{'requirement_id': 'mugs', 'status': 'satisfied',
            'evidence': 'Current external image shows mugs released on the intended right side.',
            'observation_refs': context(32)['observation_refs']}]
        state = update_planning(result, {**context(32), 'planning_state': state})
        old = deepcopy(state)
        # Many observations later: no repeated completion update is needed.
        result = action(step=1600)
        result['next_plan'] = ['Find the remaining bowl in the neighboring cabinet.']
        state = update_planning(result, {**context(1600, 4000), 'planning_state': state})
        self.assertEqual(state['task_model'], old['task_model'])
        self.assertEqual(state['progress'][0], old['progress'][0])
        self.assertEqual(state['progress'][1]['status'], 'pending')
        self.assertEqual(state['next_plan'], result['next_plan'])
        self.assertEqual(old['next_plan'], action(step=32)['next_plan'])

    def test_new_evidence_can_reopen_a_completed_claim_without_a_recovery_gate(self):
        state = update_planning(initial(), context())
        for step, status in ((32, 'satisfied'), (64, 'uncertain'), (96, 'in_progress')):
            result = action(step=step)
            result['progress_updates'] = [{'requirement_id': 'mugs', 'status': status,
                'evidence': 'New current image updates the earlier placement claim.',
                'observation_refs': context(step)['observation_refs']}]
            state = update_planning(result, {**context(step, 4000), 'planning_state': state})
            self.assertEqual(state['progress'][0]['status'], status)

    def test_cannot_drop_or_rewrite_original_requirements_or_invent_progress_ids(self):
        state = update_planning(initial(), context())
        result = action(step=32)
        for change in ({'task_model': initial()['task_model']}, {'next_plan': []},
                       {'progress_updates': [{'requirement_id': 'invented', 'status': 'satisfied',
                         'evidence': 'Not an original outcome.',
                         'observation_refs': context(32)['observation_refs']}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                update_planning({**result, **change}, {**context(32), 'planning_state': state})

    def test_progress_updates_require_current_not_stale_or_invented_evidence(self):
        state = update_planning(initial(), context())
        for refs in ([], context()['observation_refs'], [{'step': 32, 'camera': 'current:invented'}]):
            result = action(step=32)
            result['progress_updates'] = [{'requirement_id': 'mugs', 'status': 'satisfied',
                'evidence': 'Claim needs dated current evidence.', 'observation_refs': refs}]
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                update_planning(result, {**context(32), 'planning_state': state})

    def test_next_request_receives_full_state_without_reintroducing_stages(self):
        state = update_planning(initial(), context())
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append((content, kwargs))
                return action(step=32), {'model': model}

        reply = Supervisor(API(), 'muse').handle('act', {**context(32), 'planning_state': state}, {
            'current:video.robot0_agentview_left': np.zeros((8, 8, 3), dtype=np.uint8)})
        self.assertIsNone(reply['validation_error'])
        sent = json.loads(calls[0][0][0]['text'])['context']
        self.assertEqual(sent['planning_state'], state)
        self.assertNotIn('step', sent)
        self.assertEqual(len(calls), 1)
        validate_situational(action(step=32), {**context(32), 'planning_state': state})
