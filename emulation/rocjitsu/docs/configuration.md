# Configuration

Rocjitsu behavior is configured declaratively in JSON. Simulator configs
specify the component hierarchy, link connectivity, and simulation parameters;
DBT guest configs select the guest and host targets and execution backend.

## Simulator configs

Pre-built simulator configs are in `configs/`:

| File | Description |
|---|---|
| `gfx90a_mi210_kmd.json` | Single CDNA2 GPU (daemon/KFD mode) |
| `gfx942_cdna3.json` | Single CDNA3 GPU (standalone simulation) |
| `gfx942_cdna3_kmd.json` | Single CDNA3 GPU (daemon/KFD mode) |
| `gfx950_mi355x.json` | Single CDNA4 GPU (standalone simulation) |
| `gfx950_mi355x_kmd.json` | Single CDNA4 GPU (daemon/KFD mode) |
| `gfx950_mi355x_kmd_2gpu.json` | Two CDNA4 GPUs (multi-GPU daemon mode) |
| `gfx1250_mi455x.json` | Single CDNA5 GPU (standalone simulation, no KMD) |
| `gfx1250_mi455x_kmd_4gpu.json` | Four MI455X GPUs (multi-GPU daemon mode) |
| `gfx1100_w7900.json` | Single RDNA3 GPU (standalone simulation) |
| `gfx1151.json` | Single RDNA3.5 GPU (standalone simulation) |
| `gfx1201_r9700.json` | Single RDNA4 GPU (standalone simulation) |

## DBT guest configs

The checked-in [DBT guest-mode](rocjitsu_dbt_guest.md) configs cover hardware
and simulated host execution:

| File | Description |
|---|---|
| `guest_gfx950_on_gfx942.json` | CDNA4 guest on a CDNA3 hardware host |
| `guest_gfx950_on_simulated_gfx942.json` | CDNA4 guest on a simulated CDNA3 host |
| `guest_gfx950_on_gfx1201.json` | CDNA4 guest on an RDNA4 hardware host |

## JSON structure

The remaining sections describe simulator topology configs.

```json
{
  "max_ticks": 100000,
  "cpu_dispatch_threads": 1,
  "exec_mode": "functional",
  "vm": { "arch": "cdna4" },
  "topology": {
    "root": {
      "name": "soc", "type": "soc",
      "children": [
        { "name": "vram", "type": "gpu_memory" },
        { "name": "xcd[0:8]", "type": "xcd", "children": [...] }
      ]
    },
    "links": [
      {
        "pattern": "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*9+k]",
        "for_ranges": [
          { "var_name": "i", "start": 0, "end": 8 },
          { "var_name": "j", "start": 0, "end": 4 },
          { "var_name": "k", "start": 0, "end": 9 }
        ],
        "latency": 1, "weight": 10
      }
    ]
  }
}
```

The example above is intentionally minimal.

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `max_ticks` | int | Maximum simulation ticks (0 = unlimited) |
| `num_threads` | int | Simdojo engine partitions (one per XCD when partitioned). Omit for the default. |
| `cpu_dispatch_threads` | int | Inclusive functional dispatch width per SoC. Omitted/0 selects a preferred allocation; 1 forces serial dispatch. Clamped to per-CP CU capacity. |
| `async_helper_threads` | int | Shared async helpers per VM. Omitted/-1 selects a preferred allocation; 0 disables helpers. |
| `cpu_thread_budget` | int | Automatic selection ceiling. Omitted/0 uses `min(process CPU affinity, 32)`; a positive value overrides that ceiling. |
| `thread_allocations` | array | Preferred `num_threads` / `cpu_dispatch_threads` / `async_helper_threads` triples, selected by total execution-thread cost. |
| `exec_mode` | string | Execution mode. Use `"clocked"` for clocked execution; `"functional"` is the default/fallback. |
| `vm.arch` | string | Architecture: `cdna3`, `cdna4`, etc. |

`exec_mode` is matched literally: only the exact string `"clocked"` selects
clocked mode. If the field is omitted, set to `"functional"`, or given any
other value, the simulator runs in functional mode.

### Simulation threading

`num_threads` controls Simdojo engine partitions and their worker threads.
The value is clamped to the number of XCDs visible to the VM. With
`num_threads: 1`, all XCDs stay in one engine partition. With
`num_threads: 4` on the 8-XCD CDNA4 configs, whole XCD subtrees are assigned
round-robin to four partitions; with `num_threads: 8`, each XCD gets its own
partition. A single XCD is never split across partitions.

**Default.** Functional mode chooses a triple from the target config's
`thread_allocations` table. The pure `resolve_execution_threads()` function
selects the largest effective allocation fitting the budget, after applying
explicit knob overrides and topology limits. A budget between table entries
uses the lower entry; it does not create workers merely to exhaust the budget.
Later entries break ties. The automatic ceiling is `min(process CPU affinity,
32)`, with a minimum of one. Set `cpu_thread_budget` explicitly to allow a
larger entry. A config without a table uses serial defaults for unspecified
knobs. Clocked mode uses only engines, capped by affinity/budget and XCD count.

Explicit E, D or H values take precedence and may exceed the automatic ceiling.
`RJ_MMA_SHARED_HELPERS` remains an experimental override for H. E is clamped to
aggregate XCD count, and D to each SoC's largest per-CP CU count. Automatic H is
zero on targets without an enabled async MMA adapter. No workload inspection
is involved.

Two separate contracts constrain consumers, and only the first is about the
config file.

**Stepping requires a single partition.** `rj_vm_step()` and
`SimulationEngine::step()` both reject a multi-partition engine. Either pin
`"num_threads": 1` in the config, or set `loaded.engine_config.num_threads = 1`
on the `LoadedConfig` after `load_config()` returns and before constructing the
engine — the loader resolves the default, it does not enforce it.

**A multi-partition engine needs a partition policy.** Code that builds an
engine by hand must call `partition_topology_by_xcds()` after `set_root()` and
before `create()`, or `create()` throws "multi-threaded SimulationEngine
requires an explicit topology partition policy". `rj_vm_create()` already does
this, so this only affects direct `SimulationEngine` users.

For multi-GPU VMs, both the default and the clamp use the aggregate XCD count
across all SoCs. Partition assignment follows one global XCD ordering across
the SoCs and is deliberately locality-agnostic. For example, two 8-XCD GPUs
permit up to 16 partitions, while `num_threads: 4` assigns XCDs from both GPUs
to each partition.

`gfx950_mi355x_kmd_2gpu.json` and `gfx1250_mi455x_kmd_4gpu.json` are the
shipped configs that still pin `num_threads: 1`. Any multi-partition setting on
the 2-GPU config hangs RCCL collectives (`AllReduce`, `Broadcast`, `AllGather`,
`ReduceScatter`) with the engine workers spinning and the simulation making no
progress; point-to-point `SendRecv` is unaffected. The hang predates the default
and reproduces with as few as two partitions. The 4-GPU config keeps the pin for
the same reason, though the hang has only been characterised on the 2-GPU
config. Remove the pins once it is fixed.

Raising `num_threads` only pays off if the work reaches more than one XCD, which
is decided by `HwQueue::xcd_fanout` rather than by how the queue was created (see
*Queue ownership and XCD fan-out* in `vm-design.md`). KFD sets the flag for
compute queues, and a test can opt in when it registers a queue directly; a queue
without the flag keeps its whole grid on its owning XCD and leaves the other
partitions idle no matter how `num_threads` is set.

Setting the flag is not a guarantee that every partition gets work. The grid is
split in dispatch chunks, and a chunk is a whole cluster for a clustered
dispatch and a single workgroup otherwise, so what has to reach the XCD count is
the chunk count rather than the workgroup count: 16 workgroups in two
eight-workgroup clusters are two chunks, and on an eight-XCD SoC six XCDs take
an empty share and run nothing. Fan-out also reaches only the XCDs of the SoC
that owns the queue -- so in the two-GPU example above, one dispatch occupies at
most the partitions covering its own GPU.

#### Thread accounting and preferred allocations

For E engine threads, per-SoC inclusive dispatch widths D, and H shared helpers,
the retained execution allocation is **E + sum(D - 1) + H**. Each dispatch
submission also runs on its engine caller; concurrent XCD callers share the
SoC's worker pool. H is shared across all GPUs in the VM and created lazily
when async work first needs it. Runtime/doorbell/daemon threads are outside
this execution budget. Explicit `RJ_MMA_SHARED_HELPERS` uses the existing
process-wide experimental pool instead of the default VM-owned pool.

The measured eight-XCD presets use these granules:

| Budget | gfx950 E/D/H | gfx1250 E/D/H |
|---:|---:|---:|
| 1 | 1/1/0 | 1/1/0 |
| 2 | 1/2/0 | 1/2/0 |
| 4 | 1/3/1 | 1/4/0 |
| 8 | 2/5/2 | 2/5/2 |
| 16 | 4/9/4 | 4/9/4 |
| 24 | 8/12/5 | 8/12/5 |
| 32 | 8/17/8 | 8/17/8 |
| 64 (explicit budget) | 8/36/21 | 8/32/25 |

These tables approximate E:dispatch-workers:H = 1:2:1 while keeping engine
counts aligned with XCD count. A budget of 12 selects the eight-thread entry;
48 selects the 32-thread entry. See [the measurements](concurrent-dispatch-heuristic.md).
For multiple GPUs, selection counts every retained dispatch pool, so the same
triple costs more than on a single GPU. Actual useful parallelism also depends
on work reaching those GPUs/XCDs and enough runnable CUs being available.

Print the allocations for any target without constructing a simulated GPU:

```sh
rocjitsu --config configs/gfx950_mi355x.json --thread-budget-table
```

The configured row reflects the file's budget and current affinity. Remaining
rows show explicit budget ceilings while retaining the file's knob overrides.
The total column reports actual allocation, which may be below the ceiling or
above it when explicitly overridden.

For example, a deliberately small custom target can stop its preferred table
at four threads, even on a larger host:

```json
"thread_allocations": [
  {"num_threads": 1, "cpu_dispatch_threads": 1, "async_helper_threads": 0},
  {"num_threads": 1, "cpu_dispatch_threads": 2, "async_helper_threads": 0},
  {"num_threads": 1, "cpu_dispatch_threads": 4, "async_helper_threads": 0}
]
```

Mirage copies these tables from its agent configuration and leaves allocation
to rocjitsu. Profile options may override `cpu_thread_budget`, `num_threads`,
`cpu_dispatch_threads` and `async_helper_threads`; a supplied config file is
used verbatim. Checkpoints retain requests and preferred tables, so restore
re-evaluates automatic selection for the receiving process's affinity.

### Topology

Components are defined hierarchically under `topology.root`. Range
expansion (`xcd[0:8]`) creates multiple instances. Links connect
component ports using pattern expressions with loop variables.

### KFD device sections

KFD device identity can be defined by `vm.gpu.device` for a simulated GPU and
by `dbt_guest.guest_device` for a DBT guest. These sections define properties
reported through the simulated sysfs topology (GPU ID, vendor/device IDs, CU
counts, memory sizes, etc.). A simulated device's properties must match the
component hierarchy defined in `topology`.

In either device section, a device with one or more regular SDMA engines must
explicitly set a nonzero `num_sdma_queues_per_engine` value.

## FlatBuffers schema

The JSON config is validated against FlatBuffers schemas in `schemas/`:

- `simulation_config.fbs` — topology and simulation parameters
- `checkpoint.fbs` — simulation state checkpointing

## Multi-GPU

Multi-GPU configs define multiple SoCs with distinct GPU IDs and
location IDs. Each GPU gets its own command processor, memory, and
cache hierarchy. The daemon manages all GPUs and routes KFD ioctls
to the correct device based on `gpu_id`.

`configs/gfx950_mi355x_kmd_2gpu.json` is the default multi-GPU
configuration for RCCL tests. `configs/gfx1250_mi455x_kmd_4gpu.json`
provides a four-GPU daemon topology for explicit multi-GPU runs.
