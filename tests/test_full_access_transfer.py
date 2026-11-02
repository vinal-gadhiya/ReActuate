"""Full-access and transfer contracts; execute only in Thor containers."""

from copy import deepcopy
import unittest

from core.boundary_review import OPENING_CONFIRMATION_SCHEMA, opening_stage, validate_confirmation
from core.atomic_language import AtomicInstructionGrounder
from core.execution import validate_stage_switch
from core.goals import GoalLedger
from inference.supervisor import Supervisor
from tests import test_boundary_review as fixtures
from tests.test_agent import FakeTools, decision, step
from tests.test_goals import definitions, proposal, refs
from sim.run_agent import run_subtasks


class FullAccessTransferTests(unittest.TestCase):
    def context(self, instruction='Open the cabinet doors.'):
        context = fixtures.BoundaryReviewTests().context()
        context['step']['instruction'] = instruction
        context['step']['success_criteria'] = 'Both doors open with usable access.'
        return context

    def test_each_partner_needs_current_open_evidence_before_confirmation(self):
        for state in ('closed', 'partial', 'unknown'):
            result = fixtures.opening_confirmation(('open', state))
            with self.subTest(state=state), self.assertRaises(ValueError):
                validate_confirmation(result, self.context())
        for result in (fixtures.opening_confirmation(('open',)),
                       fixtures.opening_confirmation(('open',), 'single'),
                       fixtures.opening_confirmation((), 'unknown')):
            with self.assertRaises(ValueError):
                validate_confirmation(result, self.context())
        result = fixtures.opening_confirmation()
        self.assertEqual(validate_confirmation(result, self.context())['verdict'], 'confirmed')
        result['opening']['panels'][1]['observation_refs'][0]['step'] = 0
        with self.assertRaises(ValueError):
            validate_confirmation(result, self.context())

    def test_single_door_and_single_drawer_do_not_require_invented_partners(self):
        for instruction in ('Open the microwave door.', 'Fully open the drawer.'):
            context = self.context(instruction)
            context['step']['success_criteria'] = 'Named component fully open with usable clearance.'
            result = fixtures.opening_confirmation(('open',), 'single')
            self.assertTrue(opening_stage(context))
            self.assertEqual(validate_confirmation(result, context)['verdict'], 'confirmed')

    def test_duplicate_panels_cannot_masquerade_as_two_open_doors(self):
        result = fixtures.opening_confirmation()
        result['opening']['panels'][1] = deepcopy(result['opening']['panels'][0])
        with self.assertRaises(ValueError):
            validate_confirmation(result, self.context())

    def test_false_one_door_completion_is_stopped_in_supervisor_without_motion(self):
        for states, expected in ((('open', 'closed'), 'inspect'), (('open', 'open'), 'complete')):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        return {'assessment': fixtures.assessment()}, {'model': model}
                    return fixtures.opening_confirmation(states), {'model': model}

            reply = Supervisor(API(), 'muse').handle('decide', self.context(),
                fixtures.BoundaryReviewTests().images())
            self.assertEqual(reply['result']['status'], expected)
            self.assertEqual(reply['result']['program'], '')
            self.assertEqual(reply['api_calls'], 2)
            self.assertEqual(calls[1]['schema'], OPENING_CONFIRMATION_SCHEMA)

    def test_transfer_is_one_preserved_executor_request(self):
        instruction = 'Pick the mug from the counter and place it on the tray.'
        grounder = AtomicInstructionGrounder()
        grounder.validate_generated(instruction)
        responses = iter([decision(program=f'run_groot(instruction={instruction!r}, steps=4)'),
                          decision('stop', '')])
        tools, events = FakeTools(), []
        run_subtasks(tools, [step('transfer', instruction)], 'agent', lambda *args: next(responses),
                     events.append, instruction_grounder=grounder)
        self.assertEqual(tools.instructions, [instruction])
        self.assertEqual(tools.steps, 4)

    def test_decomposition_requires_current_evidence_backed_recovery(self):
        ledger = GoalLedger('Original requirements', [definitions()[0]])
        ledger.observe(0, ['current:camera'])
        ledger.observe(32, ['current:camera'])
        transfer = 'Pick the mug from the counter and place it on the tray.'
        following = 'Navigate to the dining counter.'
        result = proposal('continue', 32, following)
        with self.assertRaises(ValueError):
            validate_stage_switch(transfer, following, result, 32, ledger)
        result['recovery'] = {'necessity': 'execution_failure', 'change_type': 'strategy',
            'description': 'Observed wrong transport; navigate with retained object, then place on original tray.',
            'observation_refs': refs(0)}
        with self.assertRaises(ValueError):
            validate_stage_switch(transfer, following, result, 32, ledger)
        result['recovery']['observation_refs'] = refs(32)
        validate_stage_switch(transfer, following, result, 32, ledger)


if __name__ == '__main__':
    unittest.main()
