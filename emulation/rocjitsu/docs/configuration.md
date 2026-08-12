# Configuration

Simulation topologies are defined declaratively in JSON. The config
specifies the component hierarchy, link connectivity, and simulation
parameters.

## Config files

Pre-built configs are in `configs/`:

| File | Description |
|---|---|
| `gfx950_cdna4.json` | Single CDNA4 GPU (standalone simulation) |
| `gfx950_cdna4_kmd.json` | Single CDNA4 GPU (daemon/KFD mode) |
| `gfx950_cdna4_kmd_2gpu.json` | Two CDNA4 GPUs (multi-GPU daemon mode) |
| `gfx942_cdna3.json` | Single CDNA3 GPU (standalone simulation) |
| `gfx942_cdna3_kmd.json` | Single CDNA3 GPU (daemon/KFD mode) |
| `gfx1250.json` | Single gfx1250 GPU (standalone simulation, no KMD) |

## JSON structure

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
        "pattern": "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*8+k]",
        "for_ranges": [
          { "var_name": "i", "start": 0, "end": 8 },
          { "var_name": "j", "start": 0, "end": 4 },
          { "var_name": "k", "start": 0, "end": 8 }
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
| `cpu_dispatch_threads` | int | CPU CU-dispatch worker-pool size per command processor in functional mode (0 = auto: hardware threads, capped at 32; 1 = serial) |
| `soc_dispatch` | bool | Consolidate cross-XCD dispatch onto each SoC's primary CP for single-stream multi-XCD work distribution; requires `num_threads: 1` |
| `exec_mode` | string | `"functional"` or `"cycle"` |
| `vm.arch` | string | Architecture: `cdna3`, `cdna4`, etc. |

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

`cpu_dispatch_threads` controls how much accepted CU work can execute in
parallel on host threads; it does not change which SPI or CU accepts the
next workgroup.

`soc_dispatch` controls which command processor (CP) accepts queues for
a SoC. With `false`, queues are assigned round-robin across XCD CPs, so
each dispatch is limited to the CUs visible to the CP on its assigned
XCD. With `true`, all queues use the primary CP (`xcd[0].cp`). During
initialization, the primary CP is given the other XCDs' SPIs, CUs, and
L2 caches, so a single large dispatch can spread workgroups across all
XCDs while each CU still uses its own XCD-local L2 path.

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

See `configs/gfx950_cdna4_kmd_2gpu.json` for a working two-GPU
configuration used by the RCCL collective tests.
