"""Awomo reference-language experiment; run in Thor containers."""

import unittest
import numpy as np

from core.relational_language import instruction_grounder, instruction_policy
from core.atomic_language import AtomicInstructionGrounder
from core.boundary_review import placement_stage
from core.execution import validate_stage_switch
from core.goals import GoalLedger
from inference.supervisor import Supervisor, SUPERVISOR_POLICY
from tests.test_goals import proposal, stage, definitions, refs


class RelationalLanguageTests(unittest.TestCase):
    def test_awomo_accepts_grounded_spatial_action_frames_without_rewriting(self):
        grounder = instruction_grounder('awomo-0.5i')
        instructions = (
            'Pick the mug to the right of the brown mug.',
            'Place the mug near the brown mug.',
            'Put the mug to the left of the brown mug.',
            'Set the mug on the left side of the brown mug.',
            'Take the mug from the neighboring cabinet and place it beside the brown mug.',
            'Pick the mug from the cabinet beside the sink and place it on the neighboring counter.',
        )
        for text in instructions:
            with self.subTest(text=text):
                match = grounder.validate_generated(text)
                self.assertEqual(match['instruction_mode'], 'relational')
                grounder.validate_generated_plan([stage('action', 'first', text)])
        self.assertTrue(placement_stage({'step': {'instruction': instructions[3]}}))
        self.assertTrue(placement_stage({'step': {'instruction': instructions[4]}}))

    def test_atomic_mode_and_other_backends_retain_existing_constraints(self):
        for backend in ('patchwam', 'gr00t'):
            self.assertIsInstance(instruction_grounder(backend), AtomicInstructionGrounder)
            self.assertEqual(instruction_grounder(backend).mode, 'atomic')
            with self.assertRaises(ValueError):
                instruction_grounder(backend).validate_generated('Place the mug near the brown mug.')
        with self.assertRaises(ValueError):
            instruction_grounder('patchwam', 'relational')
        atomic = instruction_grounder('awomo-0.5i', 'atomic')
        with self.assertRaises(ValueError):
            atomic.validate_generated_plan([stage('transfer', 'first', 'Pick the mug from the counter and place it on the tray.')])

    def test_action_chains_and_hand_release_synonyms_are_not_a_bypass(self):
        grounder = instruction_grounder('awomo-0.5i')
        for text in ('Pick the mug and open the cabinet.',
                     'Pick the mug, open the cabinet.',
                     'Take the mug and place it near the bowl then close the cabinet.',
                     'Pick the mug and set it near the bowl and navigate to the sink.',
                     'Open the robot fingers.', 'Release the gripper.',
                     'Place the mug near the bowl; open the gripper.',
                     'Think carefully and do whatever is needed.'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                grounder.validate(text)
        self.assertEqual(grounder.validate('Open the gripper.')['form'], 'controlled_gripper_release')
        with self.assertRaises(ValueError):
            grounder.validate_generated_plan([stage('release', 'first', 'Open the gripper.')])

    def test_prompt_changes_language_contract_but_keeps_release_and_physical_checks(self):
        relational = instruction_policy(SUPERVISOR_POLICY, instruction_grounder('awomo-0.5i'))
        self.assertIn('RELATIONAL INSTRUCTION MODE', relational)
        self.assertNotIn('ATOMIC LANGUAGE CONTRACT:', relational)
        self.assertNotIn('Preserve their action vocabulary and sentence format', relational)
        self.assertIn('controlled_release', relational)
        self.assertIn('BOTH doors', relational)
        self.assertEqual(instruction_policy(SUPERVISOR_POLICY, instruction_grounder('patchwam')), SUPERVISOR_POLICY)

    def test_reference_refinement_requires_current_evidence_and_relational_mode(self):
        ledger = GoalLedger('Original goal', [definitions()[0]])
        ledger.observe(32, ['current:camera'])
        decision = proposal('continue', 32, 'Pick the mug beside the brown mug.')
        decision['recovery'] = {'necessity': 'reference_refinement', 'change_type': 'strategy',
            'description': 'Current view distinguishes intended mug using the same brown-mug anchor.',
            'observation_refs': refs(32)}
        with self.assertRaises(ValueError):
            validate_stage_switch('Pick up the mug.', decision['action']['instruction'], decision, 32, ledger)
        validate_stage_switch('Pick up the mug.', decision['action']['instruction'], decision, 32, ledger, allow_refinement=True)
        decision['recovery']['observation_refs'] = refs(0)
        with self.assertRaises(ValueError):
            validate_stage_switch('Pick up the mug.', decision['action']['instruction'], decision, 32, ledger, allow_refinement=True)

    def test_supervisor_uses_selected_grounder_for_executable_instruction(self):
        text = 'Put the mug near the brown mug.'

        class API:
            def response(self, *args, **kwargs):
                return proposal('continue', 0, text), {'model': 'muse'}

        context = {'control_owner': 'awomo-0.5i', 'instruction_mode': 'relational',
                   'working_context_version': 2, 'api_calls_left': 1, 'decision_step_limit': 32,
                   'capabilities': {'set_gripper': False}, 'step': stage('place', 'first', text)}
        reply = Supervisor(API(), 'muse').handle('decide', context,
                    {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertIsNone(reply['validation_error'])
        self.assertEqual(reply['result']['action']['instruction'], text)
        context['instruction_mode'] = 'atomic'
        reply = Supervisor(API(), 'muse').handle('decide', context,
                    {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertIn('not grounded', reply['validation_error'])


if __name__ == '__main__':
    unittest.main()
