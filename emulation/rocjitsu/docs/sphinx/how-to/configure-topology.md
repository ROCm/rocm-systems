---
myst:
    html_meta:
        "description": "How to select, edit, and use a JSON configuration file to define a simulated GPU topology in rocJITsu, including multi-GPU setups and C API usage."
        "keywords": "rocJITsu, ROCm, GPU topology, JSON configuration, simulation, rj_vm_create, multi-GPU, CDNA, RDNA"
---

# Configure a simulated GPU topology

rocJITsu uses a declarative JSON configuration file to describe the simulated GPU hardware: the component hierarchy, link connectivity, simulation parameters, and device properties exposed through the emulated sysfs topology. This page explains how to choose a configuration, adjust its key fields, build a multi-GPU configuration, and create a virtual machine from C code.

For a full description of the JSON schema and its FlatBuffers validation, see [JSON topology configuration](/conceptual/json-configuration.md).

## Select a configuration file

Pre-built configurations ship in the `configs/` directory:

| File | Description |
| --- | --- |
| `amdgpu_cdna4.json` | Single CDNA4 GPU (standalone simulation) |
| `amdgpu_cdna4_kmd.json` | Single CDNA4 GPU (daemon or KFD mode) |
| `amdgpu_cdna4_kmd_2gpu.json` | Two CDNA4 GPUs (multi-GPU daemon mode) |
| `amdgpu_cdna3.json` | Single CDNA3 GPU (standalone simulation) |
| `amdgpu_cdna3_kmd.json` | Single CDNA3 GPU (daemon or KFD mode) |
| `amdgpu_rdna3_gfx1100_w7900_kmd.json` | Single RDNA3 gfx1100 GPU (KFD mode) |
| `amdgpu_rdna4_gfx1201_r9700_kmd.json` | Single RDNA4 gfx1201 GPU (KFD mode) |
| `amdgpu_gfx1250.json` | Single gfx1250 GPU (standalone simulation, no KMD) |

Standalone configs (without `_kmd` in the name) are intended for caller-driven simulation where you step or run the VM directly. KMD configs initialize the emulated kernel driver so that an unmodified HIP or HSA application can issue ioctls through the LD_PRELOAD interposer.

## Edit key fields

Copy a pre-built config and modify the fields that matter for your workload. The top-level JSON structure looks like this:

``` json
{
  "max_ticks": 100000,
  "num_threads": 1,
  "exec_mode": "functional",
  "vm": { "arch": "cdna4" },
  "topology": { "..." : "..." }
}
```

### `max_ticks`

An integer that caps the number of simulation ticks. The VM stops when all primary components signal completion, when quiescence is detected, or when this limit is reached --- whichever comes first. Set to `0` for unlimited ticks (the VM runs until completion or an explicit stop request).

### `num_threads`

The number of worker threads the PDES simulation engine uses. Set to `1` for single-threaded execution.

### `exec_mode`

The execution model for compute units. Accepted values are `"functional"` (instruction-accurate, no timing) and `"cycle"` (cycle-accurate timing).

### `vm.arch`

The ISA architecture name that the VM models. This string selects the instruction decoder, execution bodies, and hardware constants. Values used in the shipped configs include `cdna3`, `cdna4`, and other architecture names matching the supported architecture table.

## KFD device section

KFD-mode configs include a `vm.gpu.device` section that defines the properties reported through the simulated sysfs topology. These properties must match the component hierarchy defined in the `topology` section. Fields in this section include GPU ID, vendor and device IDs, compute unit counts, memory sizes, cache parameters, and similar hardware descriptors.

## Build a multi-GPU configuration

Multi-GPU configs define multiple SoCs, each with a distinct GPU ID and location ID. Every GPU receives its own command processor, memory model, and cache hierarchy. The daemon manages all GPUs and routes KFD ioctls to the correct device based on `gpu_id`.

The file `configs/amdgpu_cdna4_kmd_2gpu.json` is a working two-GPU configuration used by the RCCL collective tests. To extend it:

1.  Duplicate the SoC subtree in the `topology.root.children` array.
2.  Assign each SoC a unique `gpu_id` and `location_id` in its device section.
3.  Add the corresponding link entries connecting each SoC's internal components.

## Create a VM from C code

The rocJITsu C API provides two functions for creating a virtual machine from a configuration:

-   `rj_vm_create` --- accepts a file path to a JSON config.
-   `rj_vm_create_from_string` --- accepts a JSON string directly.

Both functions take a `rj_vm_mode_t` parameter that controls initialization depth:

| Mode | Behavior |
| --- | --- |
| `RJ_VM_MODE_DEFAULT` | Standalone simulation. The caller drives execution through `rj_vm_step` or `rj_vm_run`. Topology and driver are not initialized. |
| `RJ_VM_MODE_LOCAL` | Single-process serving via LD_PRELOAD interposer. The engine runs indefinitely, and the driver is opened so that a host runtime can issue ioctls. |
| `RJ_VM_MODE_DAEMON` | Multi-process serving. Same as local mode, plus GPU allocations are backed by memfds for cross-process sharing. |

### Create from a file path

``` c
#include "rocjitsu/vm/rj_vm.h"

rj_vm_t *vm = NULL;
rj_status_t status = rj_vm_create("configs/amdgpu_cdna4_kmd.json",
                                   RJ_VM_MODE_LOCAL, &vm);
if (status != ROCJITSU_STATUS_SUCCESS) {
    fprintf(stderr, "rj_vm_create failed: %d\n", status);
    return 1;
}
```

### Create from a JSON string

``` c
const char *json = "{\"max_ticks\":100000,\"vm\":{\"arch\":\"cdna4\"}}";
rj_vm_t *vm = NULL;
rj_status_t status = rj_vm_create_from_string(json, RJ_VM_MODE_DEFAULT, &vm);
```

For the complete VM lifecycle API --- stepping, checkpointing, device commands, and memory mapping --- see [API reference: virtual machine](/reference/api-vm.md).
