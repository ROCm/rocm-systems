---
myst:
  html_meta:
    "description lang=en": "Why AMD SMI reports the same PID on multiple GPUs in multi-GPU workloads."
    "keywords": "amdsmi, amd-smi, process, pid, multi-gpu, kfd, duplicate pid, process list, compute process"
---

# Process reporting in multi-GPU workloads

When running a workload across multiple GPUs, AMD SMI may report the same PID in the process list for more than one GPU. This is expected behavior — it reflects how the ROCm kernel driver (KFD) tracks GPU compute processes.

## Why the same PID appears on multiple GPUs

AMD SMI reports processes **per GPU**. If a single process (or any of its threads) is actively using multiple GPUs simultaneously, that PID appears in the process list for each GPU it is using.

The unique identifier for a compute process in this model is the combination of **KFD node number + PID**, not PID alone. A single PID can legitimately be assigned to multiple GPUs at the same time.

This is the output structure:

```
GPU 0 → PID 12345
GPU 1 → PID 12345
GPU 2 → PID 12345
```

Each entry represents a real, active compute context on that GPU — none of these are duplicates or artifacts.

KFD reports every thread running on each GPU. This provides more granular visibility but results in the same PID appearing multiple times when a process spans GPUs.

## Multi-node workloads

On multi-node jobs, the same PID may appear on every node in the cluster. This is also expected: PID namespaces are per-node, so the same numeric PID can independently exist on each node. The unique identifier across nodes is **node hostname + KFD node number + PID**.

## Checking process details with AMD SMI

```shell
# List compute processes per GPU
amd-smi process

# Show process details for a specific GPU
amd-smi process --gpu 0

# Show JSON output for scripting
amd-smi process --json
```

## Feature request: sort by PID instead of GPU

If your workflow requires a PID-centric view (one row per unique PID with all GPUs listed), this is not currently supported.

As a workaround, you can post-process `amd-smi process --json` output to group by PID:

```python
import json, subprocess, collections

out = subprocess.check_output(["amd-smi", "process", "--json"])
data = json.loads(out)

by_pid = collections.defaultdict(list)
for gpu_entry in data:
    gpu_id = gpu_entry.get("gpu")
    for proc in gpu_entry.get("process_list", []):
        pid = proc.get("pid")
        by_pid[pid].append({"gpu": gpu_id, **proc})

for pid, entries in sorted(by_pid.items()):
    print(f"PID {pid}: GPUs {[e['gpu'] for e in entries]}")
```
