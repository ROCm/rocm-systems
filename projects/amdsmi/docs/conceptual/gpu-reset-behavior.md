---
myst:
  html_meta:
    "description lang=en": "How to isolate GPU hives for high availability when using GPU reset on XGMI systems."
    "keywords": "gpureset, xgmi, infinity fabric, pcie, kfd, fault isolation, high availability, hive, cgroups, container"
---

# GPU reset behavior on XGMI systems

On AMD Instinct systems with XGMI-connected GPUs, a reset requested for one GPU
can affect other devices depending on XGMI hive membership, PCIe topology,
active peer-to-peer traffic, and platform firmware. Do not assume that
`amd-smi reset --gpureset` provides per-GPU fault containment inside an XGMI
hive.

For high availability, use OS-enforced access controls such as containers,
cgroups, or VM passthrough to limit device access. Don't rely on
[runtime visibility variables](#xgmi-rocr-visible-devices-warning) as a
fault-isolation boundary.

## What happens during GPU reset

### Single-GPU reset

When recovery is performed for a GPU that is not part of an XGMI hive, the
driver may be able to recover only the target device. The exact recovery
sequence depends on the GPU generation, reset method, platform firmware, and
driver version.

### Hive-level effects

If the GPU belongs to an XGMI hive, a reset can have effects beyond the
selected device. Peer GPUs can be interrupted if the reset path requires XGMI
link teardown, peer notification, or recovery of shared fabric state. Active
peer-to-peer traffic can also cause peers to observe timeouts or link errors
during recovery.

Don't assume that `amd-smi reset --gpureset` provides per-GPU fault containment
inside an XGMI hive. The actual blast radius depends on the GPU
generation, reset method, driver path, firmware, PCIe topology, and workload
traffic pattern.

### XGMI link errors after reset

Persistent XGMI link errors after reset can indicate incomplete recovery. If
your platform exposes XGMI error counters through AMD SMI or sysfs, check them
before resuming workloads. See `amd-smi metric --xgmi-err`.

Because some counters are cumulative, compare against the pre-reset baseline and
watch for new or increasing errors rather than assuming all counters return to
zero.

## KFD behavior during reset

KFD manages compute queues and process GPU state via `/dev/kfd`. During
recovery, KFD can evict queues and invalidate runtime state for affected
devices. Applications should treat this as fatal for their ROCr/HIP contexts.

After recovery, applications must recreate all GPU state: queues, memory
allocations, streams, and loaded kernels. The exact set of affected processes
is driver- and platform-dependent.

(xgmi-rocr-visible-devices-warning)=

### Why `ROCR_VISIBLE_DEVICES` is not fault isolation

```{tip}
`ROCR_VISIBLE_DEVICES` is a list of device indices or UUIDs that will be
exposed to applications using user mode ROCm.
```

`ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are userspace runtime filters.
They do not create a kernel-enforced isolation boundary and do not change the
underlying reset domain managed by the kernel driver.

A process restricted with `ROCR_VISIBLE_DEVICES=0,1` can still be affected by
reset recovery outside that visible set, depending on driver behavior and
platform topology. Use OS-enforced access controls for workload isolation.

## Recommended isolation patterns

### One workload per XGMI hive

Map one workload or workload group per XGMI hive where possible. This reduces
the chance that recovery in one hive interrupts another workload, especially
when combined with OS-enforced device access controls and independent PCIe/IOMMU
topology.

### Container-based GPU access isolation

Run each workload in a separate container, exposing only its assigned GPU device
nodes. This is stronger than environment variables but weaker than VM isolation —
containers share the host kernel and KFD driver. Validate reset behavior on the
target platform.

```shell
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --device=/dev/dri/renderD129 \
  --group-add render \
  my-workload-a
```

### cgroup device-node isolation

Use cgroup device rules, directly or through a container runtime, to restrict
access to only the DRM render nodes assigned to a workload. ROCm compute
workloads generally still require access to `/dev/kfd`, so cgroups should be
treated as device-access isolation, not as a guaranteed reset-domain boundary.

This approach has lower overhead than full VM isolation, but it requires careful
device-node mapping and validation on the target platform.

### VM passthrough

Assigning a complete GPU reset domain, such as an XGMI hive, to a VM using VFIO
PCIe passthrough provides the strongest commonly available isolation because the
guest uses its own kernel and ROCm stack for the assigned devices.

Reset containment still depends on IOMMU grouping, PCIe topology, ACS support,
firmware behavior, and the reset mechanism used by the platform. Validate that a
guest-triggered reset does not affect the host or sibling VMs before relying on
this configuration for high availability.

## Handling reset events

Affected processes should not rely on partial ROCr/HIP runtime recovery after
a GPU reset. In general, they must:

1. Close all GPU handles and ROCr/HIP contexts.
2. Reopen `/dev/kfd` (by reinitializing the ROCr/HIP runtime).
3. Reallocate device memory and reload kernel state.

Checkpointing GPU state to host memory before reset enables faster recovery
but still requires full ROCr reinitialization.

## Further reading

- [amd-smi reset command reference](/how-to/amdsmi-cli-tool.md#amd-smi-reset)
- [Using AMD SMI in a Docker container](/how-to/setup-docker-container.md)
- [amdgpu GPU recovery (Linux kernel docs)](https://docs.kernel.org/gpu/amdgpu/driver-misc.html)
- [XGMI configuration (AMD Instinct virtualization driver)](https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html)
