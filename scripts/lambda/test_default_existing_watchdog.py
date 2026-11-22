import copy
import json
import unittest
from unittest.mock import patch
import default_existing_watchdog as guard
import watchdog


class TestDefault(unittest.TestCase):
    def setUp(self):
        self.target={"id":"a"*32,"workspace_id":guard.DEFAULT,"kind":"new_default_scalarpi_existing_instance_v1",
            "name":"scalar-pi-1470c2d64499","ack_sha256":"f"*64,"initial_guard_run":"123",
            "ack_unix":1795334000,"gpus":2,"type":"gpu_2x_h100_sxm5","region":"us-west-1","first_healthy":"date",
            "tags":{"project":"scalar-pi","watchdog-run":"123-1"}}
    def env(self):
        return {"DEFAULT_TARGET":json.dumps(self.target),"TARGET_SHA256":watchdog.sha256(self.target),"INSTANCE_ID":self.target["id"]}
    def test_canonical_new_gpu(self):
        self.assertEqual(guard.target_from_env(self.env()),self.target)
    def test_old_unrelated_incumbent_and_other_workspace_rejected(self):
        for iid in guard.EXCLUDED:
            self.target["id"]=iid
            with self.assertRaises(ValueError):guard.target_from_env(self.env())
        self.target["id"]="d"*32
        self.target["workspace_id"]="009e049389124c2a8e2dd8f6963d0562"
        with self.assertRaises(ValueError):guard.target_from_env(self.env())
    def test_real_shape_and_boot_provider_identity(self):
        t=self.target
        actual={"id":t["id"],"workspace_id":t["workspace_id"],"name":t["name"],"region":{"name":t["region"]},
                "instance_type":{"name":t["type"],"specs":{"gpus":2}},"first_healthy":t["first_healthy"],
                "tags":[{"key":k,"value":v} for k,v in t["tags"].items()]}
        guard.verify_identity(actual,t)
        for field in ("name","first_healthy","tags"):
            broken=copy.deepcopy(actual);broken[field]=[] if field=="tags" else "wrong"
            with self.assertRaises(ValueError):guard.verify_identity(broken,t)
    def test_different_private_pin_rejects(self):
        env=self.env();env["TARGET_SHA256"]="f"*64
        with self.assertRaises(ValueError):guard.target_from_env(env)


if __name__=="__main__":unittest.main()
