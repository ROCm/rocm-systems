---
myst:
    html_meta:
        "description": "Walk through compiling a HIP vector_add kernel for gfx950 and running it under rocJITsu in local simulation mode."
        "keywords": "rocJITsu, ROCm, GPU simulation, HIP, vector_add, tutorial, rj_vm_create, rj_vm_step, LD_PRELOAD"
---
# Simulate a HIP vector_add kernel

This tutorial walks through compiling a HIP `vector_add` kernel for the
gfx950 target and running it under rocJITsu in local (in-process)
simulation mode --- no physical GPU required. You will see the full
lifecycle from configuration loading through ISA execution and kernel
completion, and optionally step through the simulation tick by tick with
the `rj_vm_step` C API.

## Prerequisites

-   rocJITsu built from source (see [Install and build rocJITsu](../install/install.md)).
-   A ROCm toolchain that provides `hipcc` (needed to compile the kernel
    for `--offload-arch=gfx950`).

## Build rocJITsu

``` bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces the `rocjitsu` CLI at `build/tools/rocjitsu/rocjitsu`.

## Compile the HIP kernel

Write a minimal `vector_add.hip` file that launches a single-block
kernel adding two arrays element-wise, then compile it for gfx950:

``` bash
hipcc -o /tmp/vector_add vector_add.hip --offload-arch=gfx950
```

```{note}
The `--offload-arch` value must match the architecture declared in the
JSON configuration file you use to create the virtual machine. The
pre-built `configs/amdgpu_cdna4_kmd.json` targets CDNA4 (gfx950).
```

## Run the kernel in local mode

Local mode sets `LD_PRELOAD` on the target binary so that every call the
HIP runtime makes to `/dev/kfd` is intercepted and routed to the
in-process simulation engine:

``` bash
build/tools/rocjitsu/rocjitsu \
  --config configs/amdgpu_cdna4_kmd.json \
  -- /tmp/vector_add
```

The `--config` flag points to a JSON topology file that describes the
simulated GPU. The `--` separator marks the start of the application
command line. For details on the JSON format, see
[JSON topology configuration](../conceptual/json-configuration.md).

## What happens inside

The following sequence occurs when the CLI launches the application:

1.  **Configuration load** --- the JSON file is parsed against the
    embedded FlatBuffers schema and the full component hierarchy (SoC,
    XCDs, shader engines, compute units, caches, memory) is constructed.
2.  **VM creation** --- `rj_vm_create` is called with mode
    `RJ_VM_MODE_LOCAL`. This builds the simulation engine, generates the
    sysfs topology tree, and opens the simulated KFD driver so that the
    host runtime can issue ioctls through the interposer.
3.  **LD_PRELOAD interception** --- the CLI sets `LD_PRELOAD` to the
    interposer shared library and `execve`s the target application. All
    `open`, `ioctl`, `mmap`, and related syscalls targeting `/dev/kfd`
    are redirected to the simulated driver.
4.  **AQL dispatch** --- the HIP runtime submits an AQL kernel dispatch
    packet to the simulated command processor via the doorbell
    mechanism. The command processor fetches the packet from the ring
    buffer, parses the kernel descriptor, and dispatches workgroups to
    the simulated compute units.
5.  **ISA execution** --- each compute unit decodes and executes gfx950
    instructions from the kernel's `.text` section. Execution proceeds
    in the simulation engine's event-driven loop.
6.  **Completion** --- when all wavefronts retire, the completion
    tracker fires the dispatch's completion signal. The HIP runtime's
    `hipDeviceSynchronize` returns, and the application prints its
    results.

For a deeper look at the GPU hardware model and simulation architecture,
see [GPU virtual machine design](../conceptual/gpu-vm-design.md).

## Enable VM logging

To observe simulation activity, set the `RJ_LOG_GROUPS` environment
variable before launching:

``` bash
RJ_LOG_GROUPS=vm,cp build/tools/rocjitsu/rocjitsu \
  --config configs/amdgpu_cdna4_kmd.json \
  -- /tmp/vector_add
```

This enables the `vm` and `cp` (command processor) log groups, which
print timestamped messages for doorbell rings, packet fetches, workgroup
dispatches, and signal completions.

## Step through the simulation with the C API

If you want programmatic control instead of the CLI, you can drive the
simulation from C code using the virtual machine API. The key call
sequence is:

``` c
#include "rocjitsu/vm/rj_vm.h"

rj_vm_t *vm = NULL;
rj_status_t st = rj_vm_create("configs/amdgpu_cdna4_kmd.json",
                               RJ_VM_MODE_DEFAULT, &vm);
if (st != ROCJITSU_STATUS_SUCCESS) {
    fprintf(stderr, "rj_vm_create failed: %d\n", st);
    return 1;
}

int active = 1;
while (active) {
    st = rj_vm_step(vm, &active);
    if (st != ROCJITSU_STATUS_SUCCESS)
        break;
}

rj_vm_destroy(vm);
```

`rj_vm_create` loads the JSON configuration and constructs the virtual
machine. `RJ_VM_MODE_DEFAULT` creates a standalone simulation without
the KFD driver or topology generation --- the caller drives execution
manually.

`rj_vm_step` processes all simulation events at the next timestamp,
advancing the simulation by one tick. The `active` output is non-zero as
long as any wavefront is still executing.

Alternatively, `rj_vm_run` runs the simulation to completion in a single
call:

``` c
uint64_t ticks = 0;
rj_vm_run(vm, &ticks);
```

See [API reference: virtual machine](../reference/api-vm.md) for the full
virtual machine API reference.

## Next steps

-   [Inspect and disassemble a code object with the C API](inspect-code-object.md) ---
    decode and disassemble the kernel's code object with the rocJITsu
    code object API.
-   [Translate a CDNA4 kernel to RDNA4 using rj_dbt_translate](translate-cdna4-to-rdna4.md)
    --- translate the gfx950 code object to RDNA4 using the dynamic
    binary translator.
-   [Configure a simulated GPU topology](../how-to/configure-topology.md) ---
    customize the simulated GPU topology with your own JSON
    configuration.
-   [Save and restore a simulation checkpoint](../how-to/checkpoint-restore.md) --- save
    and restore simulation state.
