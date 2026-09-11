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
| `cpu_dispatch_threads` | int | Requested functional CU-dispatch width (`1` = serial and the default when omitted; `0` = explicit automatic host-wide budget capped at 32 and split across SoCs; other values = per-SoC width). Each effective SoC width is capped at its largest per-CP CU count. |
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

**Default.** Omitting `num_threads` (or setting it to `0`) selects
`min(available host threads, XCD count)` — normally one partition per XCD, since
any host that runs the simulator has more CPUs than the GPU has XCDs. "Available"
is the process's CPU affinity mask, not the machine's core count, so a job
confined to one CPU by a cgroup, a container, or `taskset` gets one partition
rather than eight. The cap keeps the simulation from asking for more workers
than it can actually run concurrently; the conservative PDES barrier makes
oversubscription markedly worse than a smaller partition count. The shipped
configs omit the field and take this default. Set it explicitly to pin a count.

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

`cpu_dispatch_threads` controls how much accepted CU work can execute in
parallel on host threads. The default value, 1, keeps dispatch serial. A
nonzero value is applied to every SoC and shared by all command processors
within that SoC. Setting the field explicitly to 0 selects one host-wide
automatic budget based on the available hardware threads, capped at 32, and
divides it as evenly as possible across the SoCs. If there are fewer available
threads than SoCs, each SoC remains serial. After either selection, each SoC's
effective width is capped at the largest number of CUs owned by any one of its
command processors, so the pool does not create workers that cannot run
additional CU tasks. This setting does not change queue ownership, XCD fan-out,
or which SPI or CU accepts the next workgroup. In clocked mode the effective
value is always 1.

#### Worked CPU-concurrency examples

The tables below separate the inputs that determine concurrency from the
resulting host-thread limits. They use current shipped topologies. `S` is the
SoC count, `X` is the XCD count per SoC, and `K` is the number of CUs reachable
from one command processor. Each listed topology has one command processor per
XCD, so `K` is also its CUs per XCD. `N` is `num_threads`, `D` is
`cpu_dispatch_threads`, `H` is the host value reported by
`hardware_concurrency()`, and `A` is the automatic-mode cap.
The no-argument runtime helper uses `A = 32`; embedding callers can supply a
different cap, but `A` is not a JSON setting. The conservative default of 32
bounds persistent worker allocation on large hosts while retaining substantial
CU parallelism. It is a policy limit, not a hardware limit.

| Case | Shipped topology | S | X | CPs/SoC | CUs/CP (`K`) | Mode | N | D | H | A |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| A | CDNA4, one GPU | 1 | 8 | 8 | 36 | functional | 8 | 1 (omitted) | 64 | 32 |
| B | CDNA4, one GPU | 1 | 8 | 8 | 36 | functional | 8 | 0 | 64 | 32 |
| C | CDNA4, one GPU | 1 | 8 | 8 | 36 | functional | 8 | 64 | 16 | 32 |
| D | CDNA4, two GPUs | 2 | 8 | 8 | 36 | functional | 1 | 0 | 64 | 32 |
| E | CDNA4, two GPUs | 2 | 8 | 8 | 36 | functional | 16 | 0 | 64 | 32 |
| F | CDNA5, four GPUs | 4 | 8 | 8 | 32 | functional | 16 | 0 | 128 | 32 |
| G | CDNA5, four GPUs | 4 | 8 | 8 | 32 | functional | 4 | 20 | 8 | 32 |
| H | CDNA4, one GPU | 1 | 8 | 8 | 36 | clocked | 8 | 0 | 64 | 32 |

`E` below is the effective Simdojo engine-thread count after clamping `N` to
the total XCD count. The dispatch policy first produces a per-SoC budget `B`,
then computes `W = min(B, K)`. At runtime, one CP batch uses at most
`min(W, runnable CUs owned by that CP)` threads. Total CUs across the SoC do not
increase `W` because its CPs share one serializing pool. `P` is the total number
of retained pool workers, `sum(W - 1)`. `T = E + P` counts execution-thread
slots, including the caller in single-threaded engine mode but excluding
doorbell monitors, daemon threads, and other runtime threads. `R` is the
maximum of those slots that can be runnable on useful simulation work at once.
`Q` counts only threads advancing CU quanta. All maxima assume enough runnable
work and favorable partition placement.

| Case | E: engine threads | W: CU threads per active CP | P: retained pool workers | T: execution threads | R: max runnable execution threads | Q: max concurrent CU quanta | Physical ceiling `min(H, R)` |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 8 | `[1]` | 0 | 8 | 8 | 8 | 8 |
| B | 8 | `[32]` | 31 | 39 | 39 | 32 | 39 |
| C | 8 | `[36]` | 35 | 43 | 43 | 36 | 16 |
| D | 1 | `[16, 16]` | 30 | 31 | 16 | 16 | 16 |
| E | 16 | `[16, 16]` | 30 | 46 | 46 | 32 | 46 |
| F | 16 | `[8, 8, 8, 8]` | 28 | 44 | 44 | 32 | 44 |
| G | 4 | `[20, 20, 20, 20]` | 76 | 80 | 80 | 80 | 8 |
| H | 8 | `[1]` | 0 | 8 | 8 | 8 | 8 |

Width `1` uses no pool, so independent XCD partitions can each advance one CU
on their engine thread. A width greater than one creates one pool per SoC; the
width includes the engine thread submitting the batch, and the pool retains
`W - 1` additional workers. Complete submissions from command processors in
the same SoC currently serialize through that pool, while different SoCs can
use their pools concurrently. This is why case D creates 31 execution-thread
slots but can use only 16 at once, and why multiplying `num_threads` by
`cpu_dispatch_threads` is not a valid concurrency formula. Cases C and G also
show that an explicit `D` bypasses the automatic cap and can oversubscribe the
host.

A checkpoint retains `D`, not `W`. Restoring `D = 0` therefore recomputes the
automatic budget from the new host's `H` and `A`; explicit requests retain
their value. Clocked mode always uses `W = 1` regardless of `D`.

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
