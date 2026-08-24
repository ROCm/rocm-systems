# Configuration

Rocjitsu behavior is configured declaratively in JSON. Simulator configs
specify the component hierarchy, link connectivity, and simulation parameters;
DBT guest configs select the guest and host targets and execution backend.

## Simulator configs

Pre-built simulator configs are in `configs/`:

| File | Description |
|---|---|
| `gfx942_cdna3.json` | Single CDNA3 GPU (standalone simulation) |
| `gfx942_cdna3_kmd.json` | Single CDNA3 GPU (daemon/KFD mode) |
| `gfx950_mi355x.json` | Single CDNA4 GPU (standalone simulation) |
| `gfx950_mi355x_kmd.json` | Single CDNA4 GPU (daemon/KFD mode) |
| `gfx950_mi355x_kmd_2gpu.json` | Two CDNA4 GPUs (multi-GPU daemon mode) |
| `gfx1250_mi455x.json` | Single CDNA5 GPU (standalone simulation, no KMD) |
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
| `num_threads` | int | Simdojo engine partitions (one per XCD when partitioned). Omit for the default |
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
`min(host hardware threads, XCD count)` — normally one partition per XCD, since
any host that runs the simulator has more threads than the GPU has XCDs. The
host cap keeps the simulation from asking for more workers than it can run
concurrently; the conservative PDES barrier makes oversubscription markedly
worse than a smaller partition count. The shipped configs omit the field and
take this default. Set it explicitly to pin a count.

Two things need `num_threads: 1` and have to say so in the config:
`rj_vm_step()` and `SimulationEngine::step()` both require a single partition,
and any code that builds an engine from a `LoadedConfig` without calling
`partition_topology_by_xcds()` will fail `create()` with "multi-threaded
SimulationEngine requires an explicit topology partition policy".

For multi-GPU VMs, both the default and the clamp use the aggregate XCD count
across all SoCs. Partition assignment follows one global XCD ordering across
the SoCs and is deliberately locality-agnostic. For example, two 8-XCD GPUs
permit up to 16 partitions, while `num_threads: 4` assigns XCDs from both GPUs
to each partition.

`gfx950_mi355x_kmd_2gpu.json` is the one shipped config that still pins
`num_threads: 1`. Any multi-partition setting on that config hangs RCCL
collectives (`AllReduce`, `Broadcast`, `AllGather`, `ReduceScatter`) with the
engine workers spinning and the simulation making no progress; point-to-point
`SendRecv` is unaffected. The hang predates the default and reproduces on the
2-GPU config with as few as two partitions. Remove the pin once it is fixed.

### Topology

Components are defined hierarchically under `topology.root`. Range
expansion (`xcd[0:8]`) creates multiple instances. Links connect
component ports using pattern expressions with loop variables.

### KFD device section

KFD-mode configs include a `vm.gpu.device` section that defines the
properties reported through the simulated sysfs topology (GPU ID,
vendor/device IDs, CU counts, memory sizes, etc.). These must match
the component hierarchy defined in `topology`.

## FlatBuffers schema

The JSON config is validated against FlatBuffers schemas in `schemas/`:

- `simulation_config.fbs` — topology and simulation parameters
- `checkpoint.fbs` — simulation state checkpointing

## Multi-GPU

Multi-GPU configs define multiple SoCs with distinct GPU IDs and
location IDs. Each GPU gets its own command processor, memory, and
cache hierarchy. The daemon manages all GPUs and routes KFD ioctls
to the correct device based on `gpu_id`.

See `configs/gfx950_mi355x_kmd_2gpu.json` for a working two-GPU
configuration used by the RCCL collective tests.
