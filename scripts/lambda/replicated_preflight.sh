#!/usr/bin/env bash
# Host inventory only. Replication, qualification and training are separate jobs.
set -euo pipefail
python3 - <<'PY'
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path('/lambda/nfs/scalar-pi')
source = Path('/var/lib/scalar-pi-replica')
attempt = os.environ['SCALAR_PI_ATTEMPT']
deadline = float(os.environ['SCALAR_PI_DEADLINE_UNIX'])
expected_count = int(os.environ['SCALAR_PI_EXPECTED_GPU_COUNT'])
assert re.fullmatch(r'scalar-pi-[a-f0-9]{12}', attempt)
assert os.geteuid() != 0


def capture(argv):
    remaining = deadline - time.time() - 15
    if remaining <= 0:
        raise TimeoutError('Host preflight reached its fixed stop window')
    return subprocess.run(argv, capture_output=True, text=True, check=True,
                          timeout=min(30, remaining)).stdout.strip()


def validate_gpus(gpu, expected_count, instance_type=None):
    assert expected_count in (1, 2, 4, 8), 'Unsupported physical GPU count'
    assert len(gpu) == expected_count, 'Physical GPU count differs from launch profile'
    rows = [line.split(', ') for line in gpu]
    if instance_type is not None:
        assert instance_type == 'gpu_1x_h100_sxm5' and expected_count == 1, 'Unexpected single H100 binding'
        families, capability, minimum_memory = ('H100',), '9.0', 80000
    elif expected_count == 1:
        families, capability, minimum_memory = ('A100-PCIE-40GB', 'A100-SXM4-40GB'), '8.0', 40000
    elif expected_count == 8:
        families, capability, minimum_memory = ('A100-SXM4-80GB',), '8.0', 80000
    else:
        families, capability, minimum_memory = ('H100',), '9.0', 80000
    assert all(len(row) == 5 and any(family in row[0] for family in families) and int(row[1]) >= minimum_memory
               and re.fullmatch(r'GPU-[a-f0-9-]{36}', row[3]) and row[4] == capability for row in rows)
    assert len({row[3] for row in rows}) == expected_count, 'Duplicate GPU UUID'
    return rows


binding_path = Path('/var/lib/scalar-pi/replicated-storage/storage-binding.json')
binding = json.loads(binding_path.read_text())
assert binding['status'] == 'bound' and binding['attempt'] == attempt
assert binding['storage_mode'] == 'replicated_local' and binding['persistence'] == 'ephemeral'
assert binding['source'] == str(source) and binding['target'] == str(root)
assert binding['deadline_unix'] == deadline and binding['qualification'] is False
assert root.resolve() == root and source.resolve() == source and os.path.samefile(source, root)
assert capture(['findmnt', '-n', '-r', '--mountpoint', str(root), '-o', 'TARGET']) == str(root)
assert capture(['git', '-C', str(root/'repo'), 'rev-parse', 'HEAD']) == 'f53255a5965153172149546dc8eee95b2bd9b076'
assert not capture(['git', '-C', str(root/'repo'), 'status', '--porcelain'])
machine = Path('/etc/machine-id').read_text().strip()
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert binding['host']['machine_id'] == machine and binding['host']['boot_id'] == boot
assert capture(['uname', '-m']) == 'x86_64'
gpu = capture(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid,compute_cap',
               '--format=csv,noheader,nounits']).splitlines()
rows = validate_gpus(gpu, expected_count, os.environ.get('SCALAR_PI_EXPECTED_INSTANCE_TYPE'))
assert not capture(['sudo', '-n', 'docker', 'ps', '--quiet'])
assert not capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
filesystem = os.statvfs(root)
free_bytes = filesystem.f_bavail * filesystem.f_frsize
minimum = expected_count * 200 * 1024**3
# Provider-advertised disk may be a separate device: inspect actual root capacity.
# Never format or move storage in response to this failure.
assert free_bytes >= minimum, 'Replica filesystem has insufficient free space; inspect mounted local storage'
minimum_ram = 180 * 1024**3 if expected_count == 1 else minimum
assert binding['host']['ram_bytes'] >= minimum_ram, 'Actual host RAM is below the profile admission minimum'
folder = root/'data/operations/hosts'/attempt/boot/'replicated-preflight'
folder.mkdir(parents=True, exist_ok=False)
record = {'schema_version': 1, 'status': 'host_observed_not_qualified', 'attempt': attempt,
          'instance_id': None, 'machine_id': machine, 'boot_id': boot, 'gpu_rows': rows,
          'expected_gpu_count': expected_count, 'physical_gpu_count': len(rows),
          'kernel': capture(['uname', '-r']), 'docker': capture(['sudo', '-n', 'docker', 'version', '--format', '{{json .}}']),
          'storage_binding': str(binding_path), 'storage_mode': 'replicated_local', 'persistence': 'ephemeral',
          'actual_free_bytes': free_bytes, 'actual_ram_bytes': binding['host']['ram_bytes'],
          'minimum_free_bytes': minimum, 'minimum_ram_bytes': minimum_ram,
          'recorded_unix': time.time(), 'deadline_unix': deadline,
          'bootstrap_commit': 'f53255a5965153172149546dc8eee95b2bd9b076',
          'scientific_qualification': False, 'inputs_copied': False, 'training_started': False}
with (folder/'preflight.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps({'status': record['status'], 'receipt': str(folder/'preflight.json')}))
PY
