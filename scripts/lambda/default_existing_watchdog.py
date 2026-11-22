"""Run one finite watchdog for a receipt-bound, newly acquired default-workspace host."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

import manual_watchdog as existing
from cloud import LambdaClient
import watchdog

DEFAULT = "c7af74c07a7e41e89f44212f1b6ecfcd"
IDENTITY = "85b2d8ab2b9b409cb8ba4050aa9c5c38"
EXCLUDED = {"d01810320c9840e9b93ea2199f0a1371","7047202517e146339f56cc87e763edf6",
            "0ebbdf650b494567ac79572e34df2afd","1df23f5628704a358ec7553b76242d89",
            "5494defd58c24c11aa110d7905551b31","48db6b1efaf5488aa3782664a6a17f18",
            "7c8a60f3e61946e88024cacc26068e8f","39650f253590490d9dbcb4cfa7c2c8f0"}
HARD = 1795420800
END = 1795413600


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def target_from_env(env):
    target=json.loads(env["DEFAULT_TARGET"])
    reference=env["TARGET_SHA256"]
    require(watchdog.sha256(target)==reference and re.fullmatch("[0-9a-f]{64}",reference), "Unrecognized private target")
    iid=target["id"]
    require(iid==env["INSTANCE_ID"] and re.fullmatch("[0-9a-f]{32}",iid) and iid not in EXCLUDED and target["workspace_id"]==DEFAULT,
            "Excluded or unbound instance/workspace")
    require(target["kind"]=="new_default_scalarpi_existing_instance_v1" and re.fullmatch(r"scalar-pi-[0-9a-f]{12}",target["name"])
            and re.fullmatch(r"[0-9a-f]{64}",target["ack_sha256"]) and str(target["initial_guard_run"]).isdigit()
            and target["ack_unix"]>=1795333970 and int(target["gpus"])>=1, "Missing actual new project launch ownership")
    return target


def verify_identity(r,t):
    tags={e["key"]:e["value"] for e in r.get("tags",[])}
    require(tags==t["tags"] and tags.get("project")=="scalar-pi" and tags.get("watchdog-run")==str(t["initial_guard_run"])+"-1",
            "Different original provider ownership tags")
    require(r["id"]==t["id"] and r["workspace_id"]==DEFAULT==t["workspace_id"] and r["name"]==t["name"]
            and r["region"]["name"]==t["region"] and r["instance_type"]["name"]==t["type"]
            and r["instance_type"]["specs"]["gpus"]==t["gpus"] and r["first_healthy"]==t["first_healthy"],
            "Original provider instance, boot, name, shape or workspace differs")


def configuration(env):
    target=target_from_env(env)
    key=env["DEFAULT_KEY"]
    audit=json.loads(env["DEFAULT_AUDIT"])
    deadline=int(env["DEADLINE_UNIX"])
    expiry=datetime.fromisoformat(audit["expires_at"].replace("Z","+00:00")).timestamp()
    require(audit["identity_id"]==IDENTITY and audit["workspace_id"]==DEFAULT and key and audit["key_sha256"]==existing.fingerprint(key)
            and sorted(audit["permission_ids"])==existing.PERMISSIONS and expiry>=deadline+watchdog.CLEANUP_SECONDS
            and deadline<=END and deadline+watchdog.CLEANUP_SECONDS<=HARD, "Same default read/terminate key and finite funded horizon required")
    return target,key,deadline


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("arm","watch"));args=parser.parse_args()
    target,key,deadline=configuration(os.environ)
    # The Derya workflow runs in a distinct process. This job reuses its unchanged bounded arm/retire implementation.
    existing.ROSTER_SHA=watchdog.sha256(target)
    existing.identity=verify_identity
    client=LambdaClient(key);path=Path("manual-watchdog/armed.json");key_hash=existing.fingerprint(key)
    if args.mode=="arm":
        proof=existing.arm(target,deadline,client,key_hash)
        require(proof["status_at_arm"]=="active","Newly acquired instance must actually be active")
        watchdog.write_json(path,proof)
    else:
        result=existing.watch(target,deadline,client,json.loads(path.read_bytes()),key_hash)
        watchdog.write_json(path.with_name("result.json"),result)


if __name__=="__main__":
    main()
