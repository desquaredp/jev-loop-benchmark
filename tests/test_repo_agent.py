import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import loop
import repo_agent as agent


class LocalBox:
    """Real git/files/process tools on a throwaway fixture; no model APIs."""
    def __init__(self, root):
        self.root = root

    def shell(self, script, stdin=None, timeout=80):
        if script.startswith('python -c '):
            script = shlex.quote(sys.executable) + script[len('python'):]
        proc = subprocess.run(script, shell=True, cwd=self.root, input=stdin,
                              text=True, capture_output=True, timeout=timeout)
        return {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}

    def diff(self):
        return loop.require_ok(self.shell('git diff --binary HEAD'))


class RepoAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.repo = root / 'repo'; self.repo.mkdir()
        self.out = root / 'out'; self.out.mkdir()
        self.box = LocalBox(self.repo)
        (self.repo / 'hidden').mkdir()
        (self.repo / 'hidden/core.py').write_text('def answer():\n    return 0\n')
        (self.repo / 'test_core.py').write_text('# existing test\n')
        for n in range(5):
            (self.repo / ('hint%d.py' % n)).write_text('# candidate %d\n' % n)
        loop.require_ok(self.box.shell('git init -q && git add . && git -c user.name=Fixture -c user.email=fixture@example.invalid commit -qm fixture'))
        self.task = {'issue': 'Fix answer', 'cheap': 'cheap', 'strong': 'strong'}
        self.guidance = {'recommended_action': 'continue_cheap', 'ranked_top_five': [
            {'path': 'hint%d.py' % n, 'rank': n + 1, 'line': 1, 'excerpt': '# hint'} for n in range(5)]}

    def do(self, action, **args):
        return agent.tool(self.box, action, args)

    def edit(self, old, new):
        return {'action': 'edit', 'args': {'edits': [{'path': 'hidden/core.py', 'old': old, 'new': new}]}, 'note': 'repair'}

    def test_repository_tools_can_find_read_and_edit_outside_priority_list(self):
        self.assertIn('hidden/core.py', self.do('files', glob='*core.py')['paths'])
        self.assertEqual(self.do('search', query='def answer')['matches'][0]['path'], 'hidden/core.py')
        self.assertIn('return 0', self.do('read', path='hidden/core.py')['content'])
        result = agent.execute(self.box, self.edit('return 0', 'return 1'))
        self.assertEqual(result['exit_code'], 0)
        self.assertIn('+    return 1', self.box.diff())

    def test_edits_are_atomic_and_existing_tests_protected(self):
        decision = self.edit('return 0', 'return 1')
        decision['args']['edits'].append({'path': 'hint0.py', 'old': 'missing text', 'new': 'oops'})
        self.assertEqual(agent.execute(self.box, decision)['status'], 'tool_error')
        self.assertEqual(self.box.diff(), '')
        result = self.do('edit', edits=[{'path': 'test_core.py', 'old': '# existing test', 'new': '# cheated'}])
        self.assertEqual(result['status'], 'tool_error')

    def test_path_escape_and_symlink_escape_blocked(self):
        outside = Path(self.tmp.name) / 'private'; outside.write_text('secret')
        (self.repo / 'escape.py').symlink_to(outside)
        for path in ('../private', str(outside), '.git/config', 'escape.py'):
            self.assertEqual(self.do('read', path=path)['status'], 'tool_error')

    def test_read_paginates_instead_of_hiding_requested_code(self):
        (self.repo / 'long.py').write_text('x\n' * 600 + 'def requested(): pass\n')
        first = self.do('read', path='long.py')
        self.assertEqual(first['next_line'], 201)
        later = self.do('read', path='long.py', start_line=601)
        self.assertIn('def requested', later['content'])

    def test_new_files_are_part_of_patch(self):
        self.do('edit', edits=[{'path': 'new_feature.py', 'old': '', 'new': 'value = 1\n'}])
        self.assertIn('new_feature.py', self.box.diff())

    def test_test_timeout_is_feedback_and_next_command_still_runs(self):
        result = self.do('test', argv=[sys.executable, '-c', 'while True: pass'], timeout_seconds=1)
        self.assertEqual(result['status'], 'test_timeout')
        self.assertEqual(result['exit_code'], 124)
        next_result = self.do('test', argv=[sys.executable, '-c', 'print("alive")'])
        self.assertEqual(next_result['exit_code'], 0)
        self.assertIn('alive', next_result['output'])

    def test_test_output_is_bounded(self):
        result = self.do('test', argv=[sys.executable, '-c', 'print("A" * 100000)'])
        self.assertLess(len(result['output']), 17000)
        self.assertIn('truncated', result['output'])

    def test_initial_control_prompts_have_no_priority_shortlist(self):
        for arm in ('cheap_only', 'strong'):
            out = self.out / arm; out.mkdir()
            captured = []
            def ask(model, payload, system):
                captured.append((model, payload, system))
                return {'action': 'finish', 'args': {}}
            result = agent.run_session(self.box, self.task, out, arm, ask,
                lambda state: self.fail('Control must not call JEV'),
                lambda directory, patch: {'status': 'failed', 'resolved': False, 'official_resolved': False}, max_turns=1)
            model, payload, system = captured[0]
            self.assertEqual(model, 'strong' if arm == 'strong' else 'cheap')
            self.assertNotIn('jev_guidance', payload)
            self.assertNotIn('JEV PRIORITIZATION', system)
            self.assertIn('search and read ANY file', system)
            self.assertFalse(result['official_resolved'])

    def test_priority_reaches_worker_and_consultant_and_timeout_is_repaired(self):
        decisions = iter([
            {'action': 'read', 'args': {'path': 'hidden/core.py'}, 'note': 'Evidence points beyond hints'},
            self.edit('return 0', 'return 1'),
            {'action': 'grade', 'args': {}},
            {'action': 'read', 'args': {'path': 'hidden/core.py'}},
            {'action': 'advise', 'args': {'diagnosis': 'wrong value', 'repair_plan': 'return 2'}},
            self.edit('return 1', 'return 2'),
            {'action': 'finish', 'args': {}},
        ])
        captured = []; route_states = []
        def ask(model, payload, system):
            captured.append((model, json.loads(json.dumps(payload)), system))
            return next(decisions)
        def route(state):
            route_states.append(json.loads(json.dumps(state)))
            return {**self.guidance, 'recommended_action': 'consult_strong' if state['failed_grades'] else 'continue_cheap'}
        def grade(directory, patch):
            if '+    return 2' in patch:
                return {'status': 'passed', 'resolved': True, 'official_resolved': True}
            return {'status': 'test_timeout', 'resolved': False, 'official_resolved': None, 'feedback': 'TIMEOUT: repair the loop'}
        result = agent.run_session(self.box, self.task, self.out, 'jev_cascade', ask, route, grade)
        self.assertTrue(result['official_resolved'])
        self.assertEqual([x[0] for x in captured], ['cheap', 'cheap', 'cheap', 'strong', 'strong', 'cheap', 'cheap'])
        for model, payload, system in captured:
            self.assertEqual(payload['jev_guidance']['ranked_top_five'], self.guidance['ranked_top_five'])
            for n in range(5): self.assertIn('%d. hint%d.py' % (n + 1, n), system)
            self.assertIn('BEFORE broad repository searches', system)
            self.assertIn('freely search/read', system)
        self.assertIn('TIMEOUT', json.dumps(captured[3][1]))
        self.assertIn('TIMEOUT', json.dumps(route_states[1]))
        self.assertEqual(captured[5][1]['specialist_advice']['repair_plan'], 'return 2')
        trace = loop.jsonl(self.out / 'trace.jsonl')
        self.assertTrue(all(len(r['jev_priority_paths_in_prompt']) == 5 for r in trace))

    def test_grading_error_preserves_actual_candidate(self):
        decisions = iter([self.edit('return 0', 'return 9'), {'action': 'grade', 'args': {}}])
        def fail(*args): raise agent.InfrastructureError('Docker disconnected')
        result = agent.run_session(self.box, self.task, self.out, 'cheap_only',
            lambda *args: next(decisions), lambda state: None, fail)
        patch_text = (self.out / 'candidate.patch').read_text()
        self.assertIn('+    return 9', patch_text)
        self.assertEqual(result['patch_sha256'], agent.sha(patch_text))
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['official_resolved'])

    def test_v2_consults_before_patch_and_returns_to_cheap(self):
        decisions=iter([{'action':'advise','args':{'repair_plan':'return 1'}},self.edit('return 0','return 1')])
        models=[]
        def ask(model,*args):
            models.append(model)
            return next(decisions)
        result=agent.run_session(self.box,{**self.task,'repair_policy_v2':True},self.out,'jev_cascade',ask,
            lambda state:{**self.guidance,'recommended_action':'consult_strong'},
            lambda *args:{'status':'passed','resolved':True,'official_resolved':True},max_turns=10)
        self.assertEqual(models,['strong','cheap'])
        self.assertTrue(result['official_resolved'])

    def test_v2_grades_early_and_feeds_failure_back_without_coding_call(self):
        decisions=iter([self.edit('return 0','return 1'),self.edit('return 1','return 2')])
        seen=[]
        def ask(model,payload,system):
            seen.append(json.loads(json.dumps(payload)))
            return next(decisions)
        def grade(directory,patch):
            passed='+    return 2' in patch
            return {'status':'passed' if passed else 'failed','resolved':passed,'official_resolved':passed,'feedback':'expected 2'}
        result=agent.run_session(self.box,{**self.task,'repair_policy_v2':True},self.out,'jev_cascade',ask,
            lambda state:self.guidance,grade,max_turns=8)
        self.assertEqual(len(seen),2)
        self.assertIn('expected 2',json.dumps(seen[1]['recent_observations']))
        self.assertTrue(result['official_resolved'])
        self.assertEqual(result['grades'],2)

    def test_repeated_action_stops_without_fourth_identical_model_call(self):
        count = []
        def ask(*args):
            count.append(1)
            return {'action': 'read', 'args': {'path': 'hidden/core.py'}}
        result = agent.run_session(self.box, self.task, self.out, 'strong', ask, lambda s: None, lambda *args: None)
        self.assertEqual(result['status'], 'stalled_repeated_action')
        self.assertEqual(len(count), 3)

    def test_missing_preflight_blocks_setup(self):
        with self.assertRaises(FileNotFoundError):
            agent.frozen_task(Path(self.tmp.name) / 'missing', self.out / 'new', 5)
        self.assertFalse((self.out / 'new').exists())

    def test_official_timeout_becomes_repair_feedback_not_infrastructure(self):
        def timed_out(source, directory, task, patch_text, label):
            folder = directory / label; folder.mkdir()
            (folder / 'test_output.txt').write_text('>>>>> Start Test Output\ntest_foo ...\nTimeout error: 180 seconds exceeded.')
            raise RuntimeError('no report')
        with patch.object(agent.hard, 'official_grade', side_effect=timed_out):
            result = agent.grade_patch(self.repo, self.out / 'grade', self.task, 'patch')
        self.assertEqual(result['status'], 'test_timeout')
        self.assertIsNone(result['official_resolved'])
        self.assertIn('nontermination', result['feedback'])

    def test_setup_timeout_stays_infrastructure(self):
        def timed_out(source, directory, task, patch_text, label):
            folder = directory / label; folder.mkdir()
            (folder / 'test_output.txt').write_text('Compiling numpy\nTimeout error: 180 seconds exceeded.')
            raise RuntimeError('no report')
        with patch.object(agent.hard, 'official_grade', side_effect=timed_out):
            with self.assertRaises(agent.InfrastructureError):
                agent.grade_patch(self.repo, self.out / 'grade', self.task, 'patch')

    def test_one_transport_retry_reuses_identical_patch_without_model_call(self):
        seen = []
        def evaluate(source, directory, task, patch_text, label):
            seen.append(patch_text)
            folder = directory / label; folder.mkdir()
            if len(seen) == 1:
                (folder / 'grader.log').write_text('500 Server Error: connection refused')
                raise RuntimeError('Docker failed')
            (folder / 'test_output.txt').write_text('all passed')
            return {'resolved': True, 'report': str(folder / 'report.json'), 'details': {'tests_status': {}}}
        with patch.object(agent.hard, 'official_grade', side_effect=evaluate):
            result = agent.grade_patch(self.repo, self.out / 'grade', self.task, 'same frozen patch')
        self.assertEqual(seen, ['same frozen patch', 'same frozen patch'])
        self.assertTrue(result['resolved'])
        self.assertEqual(result['transport_retries'], 1)

    def test_invalid_json_shape_cannot_execute_a_tool(self):
        with patch.object(agent.loop, 'call_model', return_value=(['not an action'], .01)):
            decision = agent.call_worker(object(), self.out, self.task, 'cheap_only', 'cheap', {}, 'JSON')
        self.assertEqual(decision['action'], 'invalid')

    def test_jev_ranking_is_live_and_can_include_explicitly_requested_file(self):
        state = {'note': 'Need hidden/core.py', 'observations': [], 'patch': '',
                 'failed_grades': 0, 'consulted': False}
        seen = []
        def jev(request, key=None):
            seen.append(request)
            criteria = request['questions']['primary_file']['criteria']
            return {'usage': {'input_tokens': 100}, 'answers': {
                'primary_file': {'probabilities': {k: .9 if c['path'] == 'hidden/core.py' else .01 for k, c in criteria.items()}},
                'next_action': {'probabilities': {'continue_cheap': .95, 'consult_strong': .05}}}}
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-not-a-real-key'}), patch.object(agent.spike, 'post_jev', side_effect=jev):
            guidance = agent.prioritize(self.box, self.out, {**self.task, 'budget_usd': 1}, state)
        self.assertEqual(guidance['ranked_top_five'][0]['path'], 'hidden/core.py')
        self.assertTrue(seen)
        self.assertEqual(len(loop.jsonl(self.out / 'calls.jsonl')), 1)

    def test_jev_budget_guard_prevents_call(self):
        state = {'note': 'Need hidden/core.py', 'observations': [], 'patch': '',
                 'failed_grades': 0, 'consulted': False}
        with patch.object(agent.spike, 'post_jev') as paid:
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                agent.prioritize(self.box, self.out, {**self.task, 'budget_usd': 0}, state)
            paid.assert_not_called()

    def test_early_jev_escalation_does_not_skip_cheap_investigation(self):
        models = []
        def ask(model, *args):
            models.append(model)
            return {'action': 'read', 'args': {'path': 'hidden/core.py'}}
        agent.run_session(self.box, self.task, self.out, 'jev_cascade', ask,
            lambda s: {**self.guidance, 'recommended_action': 'consult_strong'}, lambda *args: None, max_turns=1)
        self.assertEqual(models, ['cheap'])

    def test_code_change_invalidates_smoke_before_paid_calls(self):
        (self.repo / 'task.json').write_text('{}')
        loop.write_json(self.out / 'smoke.json', {'passed': True})
        loop.write_json(self.out / 'config.json', {'source_task_sha256': agent.hard.digest(self.repo / 'task.json'),
            'implementation': {name: 'old hash' for name in agent.IMPLEMENTATION}})
        with self.assertRaisesRegex(AssertionError, 'Implementation changed'):
            agent.require_smoke(self.repo, self.out)


if __name__ == '__main__':
    unittest.main()
