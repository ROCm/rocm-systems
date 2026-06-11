# Configuration

Simulation topologies are defined declaratively in JSON. The config
specifies the component hierarchy, link connectivity, and simulation
parameters.

## Config files

Pre-built configs are in `configs/`:

| File | Description |
|---|---|
| `amdgpu_cdna4.json` | Single CDNA4 GPU (standalone simulation) |
| `amdgpu_cdna4_kmd.json` | Single CDNA4 GPU (daemon/KFD mode) |
| `amdgpu_cdna4_kmd_2gpu.json` | Two CDNA4 GPUs (multi-GPU daemon mode) |
| `amdgpu_cdna3.json` | Single CDNA3 GPU (standalone simulation) |
| `amdgpu_cdna3_kmd.json` | Single CDNA3 GPU (daemon/KFD mode) |
| `amdgpu_gfx1250.json` | Single gfx1250 GPU (standalone simulation, no KMD) |

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

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `max_ticks` | int | Maximum simulation ticks (0 = unlimited) |
| `num_threads` | int | Worker threads for PDES engine |
| `exec_mode` | string | `"functional"` or `"cycle"` |
| `vm.arch` | string | Architecture: `cdna3`, `cdna4`, etc. |

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

See `configs/amdgpu_cdna4_kmd_2gpu.json` for a working two-GPU
configuration used by the RCCL collective tests.
