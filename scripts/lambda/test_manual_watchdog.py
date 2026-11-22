"""Unit tests for an existing-instance deadline guard without network access."""
import unittest
from unittest.mock import patch
import manual_watchdog as guard


class Clock:
    def __init__(self):
        self.t=100
    def now(self):
        return self.t
    def sleep(self,s):
        self.t+=s


class Client:
    def __init__(self,row,states, clock):
        self.row=row
        self.states=iter(states)
        self.posts=[]
        self.clock=clock
    def request(self,method,path,body=None):
        if method=="GET":
            status=next(self.states)
            return {**self.row,"status":status,"actions":{"terminate":{"available":status=="active"}}}
        self.posts.append((self.clock.now(),path,body))
        return {"terminated_instances":[{"id":self.row["id"]}]}


class ManualGuardTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.target={"id":"a"*32,"workspace_id":"b"*32,"region":"zone","type":"gpu","first_healthy":"BOOT"}
        self.row={**self.target,"region":{"name":"zone"},"instance_type":{"name":"gpu","specs":{"gpus":1}}}
        self.enterContext(patch.object(guard.watchdog,"github_identity",return_value={"run_id":"1"}))
        self.enterContext(patch.dict(guard.os.environ,{"WATCHDOG_JOB_STARTED_UNIX":"99"}))
    def test_deadline_confirmed_with_single_post(self):
        client=Client(self.row,["active","active","terminating","terminated"],self.clock)
        start=guard.arm(self.target,115,client,"key",now=self.clock.now)
        result=guard.watch(self.target,115,client,start,"key",now=self.clock.now,sleep=self.clock.sleep)
        self.assertEqual(result["status"],"provider_terminated")
        self.assertEqual(client.posts,[(115,"instance-operations/terminate",{"instance_ids":[self.target["id"]]})])
    def test_different_workspace_or_boot_cannot_arm(self):
        for field in ("id","workspace_id","first_healthy"):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"identity"):
                guard.arm(self.target,115,Client({**self.row,field:"different"},["active"],self.clock),"key",now=self.clock.now)
    def test_receipt_or_credential_drift_aborts_before_post(self):
        cli=Client(self.row,["active"],self.clock)
        ref=guard.arm(self.target,115,cli,"key",now=self.clock.now)
        with self.assertRaisesRegex(ValueError,"differs"):
            guard.watch(self.target,115,cli,ref,"changed",now=self.clock.now,sleep=self.clock.sleep)
        self.assertEqual(cli.posts,[])
    def test_cannot_arm_beyond_job_lifetime(self):
        with self.assertRaisesRegex(ValueError,"fit"):
            guard.arm(self.target,99999,Client(self.row,[],self.clock),"key",now=self.clock.now)
    def test_id_cannot_change_before_termination(self):
        cli=Client(self.row,["active","active"],self.clock)
        ref=guard.arm(self.target,115,cli,"key",now=self.clock.now)
        cli.row={**self.row,"id":"c"*32}
        with self.assertRaisesRegex(ValueError,"identity"):
            guard.watch(self.target,115,cli,ref,"key",now=self.clock.now,sleep=self.clock.sleep)
        self.assertEqual(cli.posts,[])

if __name__=="__main__":
    unittest.main()
