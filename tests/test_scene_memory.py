"""Memory, information-gathering guards, and temporal perception checks (container only)."""

import base64
import unittest
from unittest.mock import patch

import numpy as np

from core.agent import parse_program
from core.scene_memory import SceneMemory, validate_scene_fields
from inference.perception import tracking_records, video_part
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


class SceneMemoryTests(unittest.TestCase):
    def test_memory_preserves_identity_and_old_evidence_with_timestamps(self):
        memory = SceneMemory()
        observed = decision()
        observed['memory_updates'] = [{'name': 'lemonade glass', 'kind': 'object',
                                      'location': 'counter beside sink', 'status': 'visible',
                                      'evidence': 'Yellow drink in transparent glass in right view.'}]
        memory.update(observed, 10, 1)
        memory.update(decision(), 100, 2)
        entry = memory.context()['entries'][0]
        self.assertEqual(entry['last_seen_step'], 10)
        self.assertEqual(entry['location'], 'counter beside sink')
        self.assertEqual(len(memory.assessments), 2)
        self.assertEqual(SceneMemory().context()['entries'], [])

    def test_navigation_requires_observed_fixture_information_need_and_budget(self):
        memory = SceneMemory()
        with self.assertRaises(ValueError):
            memory.authorize_navigation('sink', 2)
        update = decision()
        update['memory_updates'] = [{'name': 'sink', 'kind': 'fixture', 'location': 'left view',
                                    'status': 'visible', 'evidence': 'Basin and faucet visible.'}]
        memory.update(update, 20, 1)
        with self.assertRaises(ValueError):
            memory.authorize_navigation('sink', 2)
        update['assessment']['information_needed'] = 'Find the destination glass.'
        memory.update(update, 20, 2)
        memory.authorize_navigation('sink', 2)
        with self.assertRaises(ValueError):
            memory.authorize_navigation('sink', 0)
        with self.assertRaises(ValueError):
            memory.authorize_navigation('glass', 2)
        self.assertEqual(parse_program('navigate(fixture="sink", steps=4)')[0][0], 'navigate')

    def test_fixture_can_be_revisited_after_leaving_current_view(self):
        memory = SceneMemory()
        observed = decision()
        observed['assessment']['information_needed'] = 'Reobserve the source area.'
        observed['memory_updates'] = [{'name': 'kitchen sink', 'kind': 'fixture', 'location': 'beside source counter',
                                      'status': 'visible', 'evidence': 'Basin and faucet visible in left view.'}]
        memory.update(observed, 0, 1)
        observed['memory_updates'][0].update(status='uncertain', location='not currently resolved',
                                             evidence='Sink no longer visible from the island.')
        memory.update(observed, 420, 2)
        memory.authorize_navigation('kitchen sink', 6)
        entry = memory.context()['entries'][0]
        self.assertEqual(entry['last_seen_step'], 0)
        self.assertEqual(entry['last_seen_location'], 'beside source counter')
        self.assertIn('Basin', entry['last_seen_evidence'])

    def test_agent_cannot_bypass_navigation_guard_but_baseline_can_run_navigation_task(self):
        tools, events = FakeTools(), []
        replies = iter([decision(program='run_groot(instruction="Navigate to the sink.", steps=4)'), decision('stop', '')])
        reason = run_subtasks(tools, [step('transfer', 'pick and place')], 'agent',
                              lambda *args: next(replies), events.append)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.steps, 0)
        self.assertTrue(any(event['event'] == 'recovery_required' for event in events))
        tools = FakeTools()
        run_subtasks(tools, [step('navigation', 'Navigate to the sink.')], 'baseline', None, lambda event: None)
        self.assertEqual(tools.steps, 10)

    def test_repeated_stationary_commands_stop_without_further_motion(self):
        class StationaryTools(FakeTools):
            last_diagnostics = {'stationary': True}

        tools, events = StationaryTools(), []
        reason = run_subtasks(tools, [step('transfer', 'pick and place')], 'agent',
                              lambda *args: decision(), events.append, subtask_budget_mode='episode', stall_bursts=2)
        self.assertEqual(reason, 'recovery_unavailable')
        self.assertEqual(tools.steps, 8)
        self.assertEqual(sum(event['event'] == 'recovery_required' for event in events), 2)

    def test_inspection_preserves_pending_goal_without_motion(self):
        tools = FakeTools()
        replies = iter([decision('inspect', ''), decision('stop', '')])
        reason = run_subtasks(tools, [step('transfer', 'pick and place')], 'agent',
                              lambda *args: next(replies), lambda event: None)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.steps, 0)
        invalid = decision()
        invalid['assessment']['grasp'] = 'supported'
        with self.assertRaises(ValueError):
            validate_scene_fields(invalid)

    def test_replan_keeps_object_location_and_grasp_evidence(self):
        memory, tools = SceneMemory(), FakeTools()
        replan = decision('replan', '', [step('new-transfer', 'place lemon in glass')])
        replan['assessment'].update(phase='carry', held_object='lemon wedge', grasp='supported',
                                    evidence='Yellow wedge moved with fingers while shelf receded.')
        replan['memory_updates'] = [{'name': 'lemonade glass', 'kind': 'object', 'location': 'counter beside sink',
                                     'status': 'visible', 'evidence': 'Glass with yellow drink visible in right view.'}]
        replies = iter([replan, decision('stop', '')])
        run_subtasks(tools, [step('transfer', 'pick lemon and place in glass')], 'agent',
                     lambda *args: next(replies), lambda event: None, scene_memory=memory)
        self.assertEqual(memory.context()['entries'][0]['location'], 'counter beside sink')
        self.assertEqual(memory.context()['recent_assessments'][0]['grasp'], 'supported')
        self.assertEqual(tools.steps, 0)

    def test_repeated_inspection_is_bounded(self):
        tools = FakeTools()
        reason = run_subtasks(tools, [step('transfer', 'place lemon')], 'agent',
                              lambda *args: decision('inspect', ''), lambda event: None)
        self.assertEqual(reason, 'perception_stall')
        self.assertEqual(tools.steps, 0)

    def test_new_destination_evidence_shortens_burst_before_first_placement_motion(self):
        tools, events = FakeTools(), []
        placing = decision(program='run_groot(instruction="Place pitcher on dining counter.", steps=20)')
        placing['assessment'].update(phase='carry', held_object='pitcher', grasp='supported', destination='dining counter')
        placing['memory_updates'] = [{'kind': 'fixture', 'name': 'dining counter', 'status': 'visible',
                                     'location': 'under held pitcher', 'evidence': 'Island top visible below pitcher.'}]
        replies = iter([placing, decision('stop', '')])
        run_subtasks(tools, [step('transfer', 'place pitcher')], 'agent', lambda *args: next(replies), events.append,
                     subtask_budget_mode='episode', placement_burst_steps=8)
        self.assertEqual(tools.steps, 8)
        self.assertTrue(any(event['event'] == 'bounded_execution' for event in events))

    def test_tracking_ids_and_frames_are_preserved_and_bad_records_rejected(self):
        text = ('<1f>7<|box;x1=1;y1=1;x2=4;y2=3;w=6;h=4|>'
                '<0f>7<|box;x1=0;y1=0;x2=3;y2=2;w=6;h=4|>'
                '<2f>7<|box;x1=0;y1=0;x2=3;y2=2;w=6;h=4|>')
        records = tracking_records(text, 6, 4, 2)
        self.assertEqual([record['frame'] for record in records], [0, 1])
        self.assertEqual([record['object_id'] for record in records], [7, 7])
        self.assertEqual(tracking_records(text, 8, 4, 2), [])

    def test_video_encoding_produces_supported_mp4(self):
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        frames[1, :, :, 0] = 255
        part = video_part(frames)
        self.assertEqual(part['type'], 'input_video')
        self.assertIn(b'ftyp', base64.b64decode(part['video_url'].split(',', 1)[1])[:32])

    def test_sam_tracking_reaches_muse_with_frame_step_mapping(self):
        calls = []

        class API:
            def response(self, model, content, instructions=None, schema=None):
                calls.append((model, content))
                if model == 'sam-3.1':
                    return '<0f>3<|box;x1=0;y1=0;x2=3;y2=2;w=6;h=4|><1f>3<|box;x1=1;y1=1;x2=4;y2=3;w=6;h=4|>', {}
                return decision(), {}

        images = {'current:wrist': np.zeros((4, 6, 3), dtype=np.uint8),
                  'clip:wrist': np.zeros((2, 4, 6, 3), dtype=np.uint8)}
        context = {'api_calls_left': 3, 'use_sam': True, 'step': step('transfer', 'pick lemon'),
                   'decision_step_limit': 40, 'capabilities': {'set_gripper': False},
                   'clip_steps': [8, 16],
                   'perception_query': {'target': 'lemon wedge', 'camera': 'current:wrist', 'mode': 'track'}}
        with patch('inference.supervisor.video_part', return_value={'type': 'input_video', 'video_url': 'test'}):
            result = Supervisor(API(), 'muse').handle('decide', context, images)
        self.assertEqual(result['api_calls'], 2)
        self.assertEqual(result['sam']['mode'], 'track')
        self.assertEqual(result['sam']['clip_steps'], [8, 16])
        self.assertEqual(len(result['sam']['tracks']), 2)
        self.assertEqual(result['sam']['boxes'][0]['frame'], 1)
        self.assertTrue(any(part['type'] == 'input_video' for part in calls[1][1]))
