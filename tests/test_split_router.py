import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import split_router as s


class SplitTests(unittest.TestCase):
    def test_failure_survives_reads_and_is_not_cleared_by_local_pass(self):
        ledger=s.Ledger()
        ledger.add({'decision':{'action':'grade'},'patch_sha256':'p1','result':{'official_resolved':True,'resolved':False,'extra_checks':{'dtype':{'exit_code':1,'output':'AssertionError'}}}})
        for _ in range(10): ledger.add({'decision':{'action':'read','args':{'path':'a.py'}},'result':{'status':'ok'}})
        ledger.add({'decision':{'action':'test'},'result':{'exit_code':0}})
        self.assertFalse(ledger.checks['dtype']['passed'])
        ledger.add({'decision':{'action':'grade'},'patch_sha256':'p2','result':{'official_resolved':True,'resolved':True,'extra_checks':{'dtype':{'exit_code':0}}}})
        self.assertTrue(ledger.checks['dtype']['passed'])

    def test_packet_excludes_patch_source_worker_claim_and_commands(self):
        ledger=s.Ledger()
        ledger.add({'decision':{'action':'test','args':{'argv':['python','PRIVATE_CODE']}},'result':{'exit_code':1,'output':'error'}})
        task={'issue':'fix','budget_usd':3,'cheap':'c','strong':'s','prices':{'c':{},'s':{}}}
        packet=ledger.packet(task,used=4,remaining=20,spent=.1,consulted=False)
        self.assertNotIn('PRIVATE_CODE',str(packet))
        self.assertEqual(packet['coding_calls_remaining'],20)
        self.assertNotIn('current_patch',packet)
        self.assertNotIn('worker_note',packet)

    def test_only_one_question_per_call(self):
        body=s.route_body({'unresolved_checks':{}})
        self.assertEqual(list(body['questions']),['next_action'])
        self.assertNotIn('Prefer cheap',body['questions']['next_action']['instructions'])

    def test_fresh_task_has_no_prior_calls_from_another_experiment(self):
        task={'issue':'fresh','budget_usd':3,'cheap':'c','strong':'s','prices':{'c':{},'s':{}}}
        with tempfile.TemporaryDirectory() as tmp:
            router=s.SplitRouter(None,Path(tmp),task,[])
            self.assertEqual(router.packet()['prior_coding_calls'],0)
            seed=[{'model':'c','decision':{'action':'read'},'result':{}}]
            resumed=s.SplitRouter(None,Path(tmp),task,seed)
            self.assertEqual(resumed.packet()['prior_coding_calls'],1)

    def test_unchanged_failed_patch_is_not_counted_twice(self):
        ledger=s.Ledger()
        event={'decision':{'action':'grade'},'patch_sha256':'same','result':{'resolved':False,'official_resolved':False}}
        ledger.add(event);ledger.add(event)
        self.assertEqual(len(ledger.checkpoints),1)

    def test_progress_counts_repeat_even_if_worker_note_changes(self):
        task={'issue':'fresh','budget_usd':3,'cheap':'c','strong':'s','prices':{'c':{},'s':{}},'repair_policy_v2':True}
        with tempfile.TemporaryDirectory() as tmp:
            router=s.SplitRouter(None,Path(tmp),task,[])
            for note in ('one','two'):
                router.ledger.add({'decision':{'action':'read','args':{'path':'a.py'},'note':note},'result':{'status':'ok'}})
            evidence=router.packet()['progress_evidence']
        self.assertEqual(evidence['investigation_calls_since_edit'],2)
        self.assertEqual(evidence['repeated_requests_since_edit'],1)

    def test_periodic_routing_happens_without_test_failures(self):
        box=type('Box',(),{'diff':lambda self:''})()
        calls=[]; routes=[]
        def ask(*args):
            calls.append(1)
            return {'action':'read','args':{'path':str(len(calls))}}
        def router(state):
            routes.append(len(calls))
            return {'recommended_action':'continue_cheap','ranked_top_five':[]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(s.agent,'tool',return_value={'status':'ok'}):
            result=s.agent.run_session(box,{'issue':'x','cheap':'c','strong':'s'},Path(tmp),'jev_cascade',
                ask,router,lambda *a:None,max_turns=9,route_every=4)
        self.assertEqual(routes,[0,4,8])
        self.assertEqual(result['status'],'turn_limit')


if __name__=='__main__':unittest.main()
