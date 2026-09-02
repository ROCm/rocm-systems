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
| `gfx1251_synthetic.json` | Minimal synthetic gfx1251 topology for functional simulation; not a product model |
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
  "num_threads": 1,
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

The example above is intentionally minimal and single-threaded.

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `max_ticks` | int | Maximum simulation ticks (0 = unlimited) |
| `num_threads` | int | Simdojo engine partitions (one per XCD when partitioned) |
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

For multi-GPU VMs, clamping uses the aggregate XCD count across all SoCs.
Partition assignment follows one global XCD ordering across the SoCs and is
deliberately locality-agnostic. For example, two 8-XCD GPUs permit up to 16
partitions, while `num_threads: 4` assigns XCDs from both GPUs to each
partition.

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
