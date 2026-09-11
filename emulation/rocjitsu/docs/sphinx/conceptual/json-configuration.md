---
myst:
    html_meta:
        "description": "JSON topology configuration for rocJITsu, covering config file structure, range expansion syntax, pattern link expressions, KFD device fields, FlatBuffers validation, and pre-built configs."
        "keywords": "rocJITsu, ROCm, JSON, topology, configuration, FlatBuffers, simulation, GPU emulation, CDNA, RDNA"
---
# JSON topology configuration

rocJITsu describes GPU topologies declaratively in JSON. A single
configuration file defines the simulation parameters, the component
hierarchy (SoC, XCDs, shader engines, compute units, caches, and
memory), and the link connectivity between components. The VM creation
functions `rj_vm_create` and `rj_vm_create_from_string` parse the JSON
against an embedded FlatBuffers schema, construct the full component
hierarchy, and load any program binaries before returning a ready-to-run
virtual machine handle.

For CMake-level build options that affect the configuration subsystem,
see [Install and build rocJITsu](../install/install.md). For a step-by-step
guide to writing or modifying a topology file, see
[Configure a simulated GPU topology](../how-to/configure-topology.md).

## Top-level fields

The root JSON object contains simulation-wide parameters and two nested
objects for the virtual machine and the topology.

``` json
{
  "max_ticks": 100000,
  "num_threads": 1,
  "cpu_dispatch_threads": 0,
  "exec_mode": "functional",
  "vm": { "arch": "cdna4" },
  "topology": {
    "root": { "..." : "..." },
    "links": [ "..." ]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `max_ticks` | int | Maximum simulation ticks. A value of `0` means unlimited. |
| `num_threads` | int | Simdojo engine partitions, clamped to the aggregate XCD count. |
| `cpu_dispatch_threads` | int | Requested functional CU-dispatch width. `1` is serial and the default when omitted; `0` explicitly selects an automatic host-wide budget capped at 32 and split across SoCs; other values apply per SoC. Each effective SoC width is capped at its largest per-CP CU count. |
| `exec_mode` | string | Execution mode: `"functional"` or `"clocked"`. |
| `vm.arch` | string | Target architecture, such as `cdna3`, `cdna4`, or `rdna4`. |

`num_threads` and `cpu_dispatch_threads` control different layers.
`num_threads` partitions whole XCD subtrees across Simdojo engine threads. A
single XCD is never split between engine partitions. In functional mode,
`cpu_dispatch_threads` controls the host parallelism used to execute accepted
CU work. Its omitted-field default is serial; set it explicitly to `0` to use
automatic sizing. Its budget is shared by all command processors in a SoC and
does not change queue ownership or XCD fan-out. After either automatic or explicit
selection, the effective width is capped at the largest number of CUs owned by
any one command processor in that SoC. In clocked mode its effective value is
always 1.

`exec_mode` is matched literally. Only `"clocked"` selects cycle-accurate
mode. If the field is omitted or set to `"functional"`, `"cycle"`, or any
other value, rocJITsu runs in functional mode.


## Component hierarchy and range expansion

Components are arranged hierarchically under `topology.root`. Each
component node has a `name`, a `type`, and an optional `children` array.

``` json
{
  "name": "soc", "type": "soc",
  "children": [
    { "name": "vram", "type": "gpu_memory" },
    { "name": "xcd[0:8]", "type": "xcd", "children": ["..."] }
  ]
}
```

The **range expansion** syntax `xcd[0:8]` creates eight instances named
`xcd0` through `xcd7`. The range is half-open: the start value is
inclusive and the end value is exclusive. You can nest ranges at
multiple levels of the hierarchy --- for example, `se[0:4]` inside each
`xcd[0:8]` --- to produce a combinatorial set of components.

## Pattern link expressions

Links connect component ports. Rather than listing every connection
individually, the configuration uses **pattern expressions** with loop
variables to wire large topologies compactly.

``` json
{
  "pattern": "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*8+k]",
  "for_ranges": [
    { "var_name": "i", "start": 0, "end": 8 },
    { "var_name": "j", "start": 0, "end": 4 },
    { "var_name": "k", "start": 0, "end": 8 }
  ],
  "latency": 1,
  "weight": 10
}
```

Each entry in the `links` array can include:

-   `pattern` --- A string expression using bracket-delimited loop
    variables. The left-hand side names the source port and the
    right-hand side names the destination port, separated by `->`.
    Arithmetic expressions such as `j*8+k` are evaluated during
    expansion.
-   `for_ranges` --- An array of range descriptors, each specifying
    `var_name`, `start` (inclusive), and `end` (exclusive). The link is
    instantiated for every combination of range values.
-   `latency` --- Simulation latency for the link.
-   `weight` --- Link weight used by the simulation scheduler.

For a topology with 8 XCDs, 4 shader engines per XCD, and 8 compute
units per shader engine, the example above produces 256 links from a
single pattern entry.

## KFD device section

Configs intended for KFD or daemon mode (`RJ_VM_MODE_LOCAL` or
`RJ_VM_MODE_DAEMON`) include a `vm.gpu.device` section. This section
defines the properties that the simulated driver exposes through the
sysfs topology interface, such as GPU ID, vendor and device IDs, compute
unit counts, memory sizes, cache parameters, and marketing name. These
values must be consistent with the component hierarchy defined in
`topology`.

The fields in this section correspond to the members of
`rj_vm_gpu_info_t` documented in [API reference: virtual machine](../reference/api-vm.md). When `rj_vm_gpu_info` is called, the returned structure
reflects the device section values from the active configuration.

## FlatBuffers schema validation

The JSON configuration is validated against FlatBuffers schemas located
in the `schemas/` directory of the repository:

-   `simulation_config.fbs` --- Covers topology and simulation
    parameters.
-   `checkpoint.fbs` --- Covers simulation state checkpointing (see
    [Save and restore a simulation checkpoint](../how-to/checkpoint-restore.md)).

Validation occurs at VM creation time. If the JSON does not conform to
the schema, `rj_vm_create` returns `ROCJITSU_STATUS_ERROR`. Common
validation failures include missing required fields, type mismatches,
and malformed range expressions.

## Pre-built configurations

The `configs/` directory ships several ready-to-use topology files:

| File | Description |
|------|-------------|
| `gfx90a_mi210_kmd.json` | Single CDNA2 GPU, daemon or KFD mode. |
| `gfx942_cdna3.json` | Single CDNA3 GPU, standalone simulation. |
| `gfx942_cdna3_kmd.json` | Single CDNA3 GPU, daemon or KFD mode. |
| `gfx950_mi355x.json` | Single CDNA4 GPU, standalone simulation. |
| `gfx950_mi355x_kmd.json` | Single CDNA4 GPU, daemon or KFD mode. |
| `gfx950_mi355x_kmd_2gpu.json` | Two CDNA4 GPUs, multi-GPU daemon mode. |
| `gfx1250_mi455x.json` | Single CDNA5 GPU, standalone simulation. |
| `gfx1250_mi455x_kmd_4gpu.json` | Four CDNA5 GPUs, multi-GPU daemon mode. |
| `gfx1100_w7900.json` | Single RDNA3 GPU, standalone simulation. |
| `gfx1151.json` | Single RDNA3.5 GPU, standalone simulation. |
| `gfx1201_r9700.json` | Single RDNA4 GPU, standalone simulation. |


Standalone configs (without `_kmd` in the name) are used with
`RJ_VM_MODE_DEFAULT`, where the caller drives the simulation via
`rj_vm_step` or `rj_vm_run`. KMD configs initialize the topology and
simulated driver so that an unmodified ROCm runtime stack can issue
ioctls through the interposer.

## Multi-GPU configurations

Multi-GPU configs define multiple SoCs, each with a distinct GPU ID and
location ID. Every GPU receives its own command processor, memory
subsystem, and cache hierarchy. In daemon mode, the simulated driver
manages all GPUs and routes KFD ioctls to the correct device based on
`gpu_id`. The `gfx950_mi355x_kmd_2gpu.json` config provides a working
two-GPU configuration used with RCCL collective tests.
