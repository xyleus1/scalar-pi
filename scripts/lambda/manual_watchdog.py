"""Arm and retire a hash-pinned, already purchased Lambda instance at its finite deadline."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from cloud import LambdaApiError, LambdaClient
import watchdog

ROSTER_SHA = "9b0bcb71f4f0e1f5be2ecb25bfa93edab51926197861c7c7c8ad7da97f5f745f"
HARD_CLEANUP = 1795420800
PERMISSIONS = ["compute:instance:read", "compute:instance:terminate"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def roster_row(text, index):
    rows = json.loads(text)
    require(watchdog.sha256(rows) == ROSTER_SHA and 0 <= index < 7, "Unrecognized manual instance roster")
    return rows[index]


def identity(row, target):
    require(row["id"] == target["id"] and row["workspace_id"] == target["workspace_id"]
            and row["region"]["name"] == target["region"] and row["instance_type"]["name"] == target["type"]
            and row["instance_type"]["specs"]["gpus"] == 1 and row["first_healthy"] == target["first_healthy"],
            "Real instance, workspace, boot identity, region or hardware changed")


def fingerprint(key):
    return hashlib.sha256(key.encode()).hexdigest()


def configuration(env):
    target = roster_row(env["DERYA_MANUAL_ROSTER"], int(env["TARGET_INDEX"]))
    deadline = int(env["DEADLINE_UNIX"])
    key = env["LAMBDA_DERYA_TERMINATE_API_KEY"]
    audit = json.loads(env["DERYA_MANUAL_KEY_AUDIT"])
    expires = datetime.fromisoformat(audit["expires_at"].replace("Z", "+00:00")).timestamp()
    require(key and audit["key_sha256"] == fingerprint(key) and audit["workspace_id"] == target["workspace_id"]
            and sorted(audit["permission_ids"]) == PERMISSIONS and expires >= deadline + watchdog.CLEANUP_SECONDS
            and deadline + watchdog.CLEANUP_SECONDS <= HARD_CLEANUP,
            "A dedicated, audited Derya read/terminate credential through cleanup is required")
    return target,deadline,key


def arm(target, deadline, client, key_hash, *, now=time.time):
    started = int(os.environ["WATCHDOG_JOB_STARTED_UNIX"])
    until = deadline + watchdog.CLEANUP_SECONDS
    require(started <= now() < until and until + watchdog.EVIDENCE_RESERVE_SECONDS <= started + watchdog.JOB_TIMEOUT_SECONDS,
            "Finite deadline and cleanup must fit this actual GitHub job")
    row = watchdog.get(client, "instances/" + target["id"], min(until, now() + 180), now=now)
    identity(row,target)
    return {"kind":"existing_lambda_instance_shutdown_armed_v1", "github":watchdog.github_identity(),
            "target_sha256":watchdog.sha256(target), "roster_sha256":ROSTER_SHA, "key_sha256":key_hash,
            "deadline_unix":deadline, "cleanup_until_unix":until, "observed_unix":now(), "status_at_arm":row["status"]}


def watch(target, deadline, client, receipt, key_hash, *, now=time.time, sleep=time.sleep):
    until = deadline + watchdog.CLEANUP_SECONDS
    require(receipt["kind"] == "existing_lambda_instance_shutdown_armed_v1" and receipt["github"] == watchdog.github_identity()
            and receipt["target_sha256"] == watchdog.sha256(target) and receipt["roster_sha256"] == ROSTER_SHA
            and receipt["deadline_unix"] == deadline and receipt["cleanup_until_unix"] == until and receipt["key_sha256"] == key_hash,
            "Run, key, target, or deadline differs from the actual armed receipt")
    while now() < deadline:
        sleep(min(60, deadline-now()))
    result = {"target_sha256":receipt["target_sha256"], "github":receipt["github"], "deadline_unix":deadline,
              "post_attempts":0, "termination_acknowledged":False}
    last_post = None
    missing = 0
    while now() < until:
        row = watchdog.get(client, "instances/"+target["id"], until, now=now, sleep=sleep, missing_ok=True)
        if row is None:
            missing += 1
            if missing >= 2:
                return {**result,"status":"absent_after_two_gets","observed_unix":now()}
        else:
            missing = 0
            identity(row,target)
            if row["status"] == "terminated":
                return {**result,"status":"provider_terminated","observed_unix":now()}
            available = row.get("actions",{}).get("terminate",{}).get("available") is True
            if (row["status"] != "terminating" and available and not result["termination_acknowledged"]
                    and result["post_attempts"] < watchdog.MAX_TERMINATE_ATTEMPTS
                    and (last_post is None or now()-last_post>=watchdog.RECONCILE_SECONDS)):
                last_post=now()
                result["post_attempts"]+=1
                try:
                    answer=client.request("POST","instance-operations/terminate",{"instance_ids":[target["id"]]})
                except (LambdaApiError, RuntimeError, TimeoutError, OSError) as error:
                    if isinstance(error,LambdaApiError) and not watchdog.transient(error):
                        raise
                else:
                    ids=[r["id"] for r in answer.get("terminated_instances",[])]
                    require(all(i==target["id"] for i in ids),"Unexpected provider termination acknowledgement")
                    result["termination_acknowledged"]=ids==[target["id"]]
        sleep(min(watchdog.POLL_SECONDS,max(0,until-now())))
    raise TimeoutError("Terminal state unconfirmed at the absolute cleanup deadline")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("operation", choices=("arm","watch"))
    args=parser.parse_args()
    target,deadline,key=configuration(os.environ)
    client=LambdaClient(key)
    receipt=Path("manual-watchdog/armed.json")
    if args.operation=="arm":
        watchdog.write_json(receipt,arm(target,deadline,client,fingerprint(key)))
    else:
        result=watch(target,deadline,client,json.loads(receipt.read_bytes()),fingerprint(key))
        watchdog.write_json(receipt.with_name("result.json"),result)


if __name__=="__main__":
    main()
