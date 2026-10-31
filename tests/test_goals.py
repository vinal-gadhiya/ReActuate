"""Task-general ledger, rolling windows, and recovery contracts; Thor containers only."""

from copy import deepcopy
import json
import unittest

from core.agent import validate_stage_program
from core.goals import GoalLedger, AttemptHistory, validate_requirements
from core.instructions import PolicyInstructionAdapter
from core.scene_memory import validate_scene_fields
from core.working_context import working_context
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


def definitions():
    return [{'id': 'first', 'description': 'Establish the first required outcome', 'dependencies': []},
            {'id': 'second', 'description': 'Establish the second required outcome', 'dependencies': ['first']}]


def refs(at):
    return [{'step': at, 'camera': 'current:camera'}]


def update(identifier, status, at):
    return {'goal_id': identifier, 'status': status, 'evidence': 'Current observation supports this judgment.',
            'observation_refs': refs(at)}


def proposal(status, at, instruction='', goal_updates=None, replacement=None):
    result = decision(status, f'run_groot(instruction={instruction!r}, steps=1)' if instruction else '', replacement)
    result.update(assessment={'outcome': 'satisfied' if status == 'complete' else 'unknown',
                              'evidence': 'Visible outcome evidence.', 'observation_refs': refs(at), 'transport': None},
                  goal_updates=goal_updates or [], attempt_reports=[], recovery=None,
                  action={'intent': 'interact', 'object': '', 'destination': '', 'instruction': instruction} if instruction else None)
    return result


def stage(identifier, goal_id, instruction='Execute the next supported action.'):
    return {**step(identifier, instruction), 'goal_ids': [goal_id]}


class GoalLedgerTests(unittest.TestCase):
    def test_dependencies_and_duplicate_requirements_are_checked_without_task_names(self):
        validate_requirements(definitions())
        for bad in ([definitions()[0], definitions()[0]],
                    [{'id': 'a', 'description': 'Outcome', 'dependencies': ['missing']}],
                    [{'id': 'a', 'description': 'Outcome', 'dependencies': ['a']}]):
            with self.assertRaises(ValueError):
                validate_requirements(bad)

    def test_only_status_and_evidence_can_change_and_reopening_needs_new_observations(self):
        ledger = GoalLedger('Original instruction', definitions())
        ledger.observe(0, ['current:camera'])
        original = ledger.definitions()
        with self.assertRaises(ValueError):
            ledger.apply_updates([update('second', 'satisfied', 0)], 0)
        ledger.apply_updates([update('first', 'satisfied', 0)], 0)
        self.assertEqual(ledger.definitions(), original)
        with self.assertRaises(ValueError):
            ledger.apply_updates([update('first', 'unsatisfied', 0)], 0)
        ledger.observe(1, ['current:camera'])
        ledger.apply_updates([update('first', 'unsatisfied', 1)], 1)
        self.assertEqual(ledger.pending(), ['first', 'second'])
        invalid = {**update('first', 'satisfied', 1), 'description': 'Erase the original goal'}
        with self.assertRaises(ValueError):
            ledger.apply_updates([invalid], 1)

    def test_unknown_future_and_unseen_camera_claims_are_rejected_atomically(self):
        ledger = GoalLedger('Original', definitions())
        ledger.observe(0, ['current:camera'])
        before = ledger.context()
        bad = update('first', 'satisfied', 1)
        with self.assertRaises(ValueError):
            ledger.apply_updates([bad], 0)
        bad = update('missing', 'satisfied', 0)
        with self.assertRaises(ValueError):
            ledger.apply_updates([update('first', 'satisfied', 0), bad], 0)
        self.assertEqual(ledger.context(), before)

    def test_action_window_does_not_need_to_contain_every_pending_requirement(self):
        ledger = GoalLedger('Original', definitions())
        ledger.validate_window([stage('short-window', 'first')])
        self.assertEqual(ledger.pending(), ['first', 'second'])
        with self.assertRaises(ValueError):
            ledger.validate_window([stage('renamed', 'invented')])
        self.assertEqual(json.loads(json.dumps(ledger.context()))['pending_goal_ids'], ['first', 'second'])

    def test_rolling_windows_extend_without_consuming_recovery_replans(self):
        ledger, tools, events = GoalLedger('Two outcomes', definitions()), FakeTools(), []
        next_stage = stage('later-window', 'second')
        replies = iter([
            proposal('continue', 0, 'Turn on the appliance.'),
            proposal('complete', 1, goal_updates=[update('first', 'satisfied', 1)]),
            proposal('replan', 1, replacement=[next_stage]),
            proposal('continue', 1, 'Heat the contents.'),
            proposal('complete', 2, goal_updates=[update('second', 'satisfied', 2)]),
        ])

        def decide(*args):
            ledger.observe(tools.steps, ['current:camera'])
            return next(replies)

        reason = run_subtasks(tools, [stage('first-window', 'first')], 'agent', decide, events.append,
                              goal_ledger=ledger, subtask_budget_mode='episode', max_replans=0)
        self.assertEqual(reason, 'requirements_reported_satisfied')
        self.assertFalse(tools.success)  # Model claims never change simulator scoring.
        self.assertEqual(tools.instructions, ['Turn on the appliance.', 'Heat the contents.'])
        self.assertEqual(ledger.pending(), [])
        replaced = next(item for item in events if item['event'] == 'plan_replaced')
        self.assertEqual(replaced['kind'], 'window_extension')
        self.assertEqual(ledger.definitions(), definitions())

    def test_nontransport_outcomes_are_first_class_for_different_action_families(self):
        for instruction in ('Wash the dish.', 'Heat the contents.', 'Turn off the fixture.',
                            'Pour the liquid into the vessel.', 'Stir the contents.'):
            result = proposal('continue', 0, instruction)
            validate_scene_fields(result)
            validate_stage_program([('run_groot', {'instruction': instruction, 'steps': 1})])
            self.assertIsNone(result['assessment']['transport'])


class AttemptHistoryTests(unittest.TestCase):
    def ledger(self):
        return GoalLedger('Required outcome', [definitions()[0]])

    def failure(self, history, outcome, before, instruction='Pick up the required item.'):
        attempt = history.record(['first'], 'run_groot', {'instruction': instruction, 'steps': 1}, None, before, before + 1)
        history.ledger.observe(before + 1, ['current:camera'])
        history.review([{'attempt_id': attempt['attempt_id'], 'outcome': outcome,
                         'evidence': 'Supported temporal observation.', 'observation_refs': refs(before + 1)}], before + 1)
        return attempt

    def test_wrong_required_object_cycles_survive_replans_and_burst_size_changes(self):
        history = AttemptHistory(self.ledger(), failure_limit=3)
        for at in range(3):
            self.failure(history, 'other_required_object_acquired', at)
        self.assertEqual(len(history.failures('first')), 3)
        with self.assertRaisesRegex(ValueError, 'changed strategy'):
            history.authorize(['first'], 'run_groot', {'instruction': 'Pick up the required item.', 'steps': 32}, None, 3)
        recovery = {'change_type': 'strategy', 'description': 'Renamed stage and retry', 'observation_refs': refs(3)}
        with self.assertRaisesRegex(ValueError, 'repeats a failed command'):
            history.authorize(['first'], 'run_groot', {'instruction': ' PICK UP THE REQUIRED ITEM. ', 'steps': 8}, recovery, 3)
        recovery['description'] = 'Use a different grounded request to correct identification'
        history.authorize(['first'], 'run_groot', {'instruction': 'Pick up the striped item.', 'steps': 8}, recovery, 3)

    def test_uncertainty_is_not_confirmed_failure_and_repeated_reports_do_not_double_count(self):
        history = AttemptHistory(self.ledger(), failure_limit=2)
        attempt = self.failure(history, 'uncertain', 0)
        self.assertEqual(history.failures('first'), [])
        report = {'attempt_id': attempt['attempt_id'], 'outcome': 'no_reported_progress',
                  'evidence': 'Supported temporal observation.', 'observation_refs': refs(1)}
        history.review([report], 1)
        history.review([report], 1)
        self.assertEqual(len(history.failures('first')), 1)

    def test_prerequisite_retry_requires_new_executed_and_observed_change(self):
        history = AttemptHistory(self.ledger(), failure_limit=1)
        self.failure(history, 'no_reported_progress', 0)
        recovery = {'change_type': 'prerequisite', 'description': 'The obstruction is now cleared', 'observation_refs': refs(2)}
        history.ledger.observe(2, ['current:camera'])
        with self.assertRaisesRegex(ValueError, 'subsequent executed action'):
            history.authorize(['first'], 'run_groot', {'instruction': 'Pick up the required item.', 'steps': 1}, recovery, 2)
        self.failure(history, 'prerequisite_changed', 1, 'Open the fixture.')
        original = deepcopy(recovery)
        arguments = {'instruction': 'Pick up the required item.', 'steps': 1}
        authorized = history.authorize(['first'], 'run_groot', arguments, recovery, 2)
        self.assertEqual(recovery, original)
        history.record(['first'], 'run_groot', arguments, None, 2, 3, authorized)
        with self.assertRaisesRegex(ValueError, 'same prerequisite evidence'):
            history.authorize(['first'], 'run_groot', arguments, recovery, 2)

    def test_recovery_limit_persists_when_execution_stage_is_replaced(self):
        history = AttemptHistory(self.ledger(), failure_limit=1, recovery_limit=1)
        self.failure(history, 'no_reported_progress', 0)
        recovery = {'change_type': 'strategy', 'description': 'Use a grounded alternate approach', 'observation_refs': refs(1)}
        arguments = {'instruction': 'Open the fixture.', 'steps': 1}
        authorization = history.authorize(['first'], 'run_groot', arguments, recovery, 1)
        history.record(['first'], 'run_groot', arguments, None, 1, 2, authorization)
        with self.assertRaisesRegex(ValueError, 'allowance exhausted'):
            history.authorize(['first'], 'run_groot', {'instruction': 'Reposition at the fixture.', 'steps': 1}, recovery, 1)


class InstructionAndContextTests(unittest.TestCase):
    def test_backend_boundary_preserves_grounded_text_and_rejects_semantic_command_mismatch(self):
        instruction = 'Open the left door of the fixture beside the visible appliance to access the item.'
        action = {'intent': 'open', 'object': 'fixture', 'destination': '', 'instruction': instruction}
        for backend in ('gr00t', 'patchwam'):
            adapter = PolicyInstructionAdapter(backend)
            self.assertEqual(adapter.render(action), instruction)
            validate_stage_program([('run_groot', {'instruction': instruction, 'steps': 1})])
            with self.assertRaises(ValueError):
                adapter.validate_command(action, 'run_groot', {'instruction': 'Do something else.', 'steps': 1})

    def test_working_context_omits_overlapping_history_and_retains_belief_age(self):
        source = {'original_goal': 'A general task', 'observation_step': 20,
                  'episode_context': {'old': 'large'}, 'scene_assessment': {'old': 'survey'},
                  'execution_feedback': {'old': 'history'}, 'history': ['old events'],
                  'scene_memory': {'entries': [{'name': 'fixture', 'kind': 'fixture', 'status': 'uncertain',
                                                'location': 'previous workspace', 'observation_step': 5}]}}
        result = working_context(source)
        self.assertEqual(result['beliefs'][0]['age_steps'], 15)
        for field in ('episode_context', 'scene_assessment', 'execution_feedback', 'history'):
            self.assertNotIn(field, result)
        self.assertIn('episode_context', source)


if __name__ == '__main__':
    unittest.main()
