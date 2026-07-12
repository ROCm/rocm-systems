# gpumetrics: a focused GPU metrics library + CLI

`gpumetrics` reads and exposes AMD GPU metrics per **GPU**, per **partition**, and per
**socket**, and nothing else. No gRPC, no daemon, no job tracking, no diagnostics. Just
metrics, from pluggable backends, behind a small stable C API with a C++ implementation and a
thin CLI.

It is a from-scratch redesign informed by a deep read of ROCm Data Center Tool (RDC). It keeps
RDC's good ideas (a backend-agnostic plugin contract with query/route/read, a compact entity
model) and fixes RDC's structural pain points: a 2000-line hand-written `switch` for metric
dispatch, a 217-value field enum with no type/unit registry, per-read topology re-enumeration,
ad-hoc cross-backend GPU correlation, and sentinel-value overloading.

## Goals / non-goals

**Goals**
- Read metrics from multiple backends (plugins). Ship two: `amdsmi` (telemetry) and
  `rocprofiler` (hardware perf counters / derived metrics).
- Address the same metric consistently regardless of which plugin serves it.
- Select a target unambiguously (physical GPU, a partition of a GPU, or a socket) even when
  two plugins index devices differently.
- Small, stable C ABI so a Rust layer can bind later.
- Test-driven; values validated against `amd-smi`.

**Non-goals**: control/set operations, gRPC/remote, kernel-dispatch profiling, CPU metrics
(the model leaves room, but v1 is GPU-focused).

## Metric model

A metric is identified by a **stable string key** in dotted-namespace form, e.g.
`temp.edge`, `power.average_socket`, `clock.gfx`, `ecc.total.correctable`, `prof.active_cycles`,
`prof.occupancy_pct`. Strings (not a giant enum) keep plugins decoupled from a central enum and
let the registry be the single source of truth.

Every metric has a descriptor (the registry entry RDC lacked): `{key, type, unit, scope,
description, provider}`. A reading pairs the key with a typed value, a status, and a timestamp.
The registry lives in the core, populated from descriptors plugins declare at load time, so the
CLI formats values with no per-field special cases.

## Entity model (GPU / partition / socket)

The addressable thing is an **Entity**, whose identity is a struct, not a packed int (RDC
packed everything into 32 bits and re-derived kind from enum ranges):

```
enum class EntityKind { Socket, Gpu, GpuPartition };
struct EntityId { EntityKind kind; uint32_t socket; uint32_t gpu; int32_t partition; };
```

Selectors accepted by the CLI/API: `gpu:0`, `g0.1` (partition 1 of GPU 0), `socket:0`,
`bdf:0000:63:00.0`, `uuid:<hex>`, a bare index, and `all`.

### Cross-plugin correlation: the hard part

Two plugins enumerate devices independently and in different orders (amdsmi walks
socket -> processor handles; rocprofiler returns a flat agent list), so the core must know that
amdsmi device X and rocprofiler agent Y are the same physical GPU. Each plugin reports a
`gpum_device_identity` per device it serves, and the core's `DeviceRegistry` groups them into
canonical GPUs.

The subtle part is **partitioning**. On MI350X in CPX mode a physical GPU splits into 8
partitions, and measurement on real hardware showed that most identity keys fragment per
partition: each partition is its own PCIe function with a distinct full BDF, KFD node id, and
UUID. Only two keys stay per-physical-GPU:

- **masked BDF** (domain:bus:device, PCIe function stripped): all partitions of one GPU share
  it, and distinct GPUs differ in bus/device.
- **oam_id**: the OAM physical-slot id, valid on the whole-GPU handle (partitions report a
  sentinel).

So the correlation key priority is **masked BDF -> oam_id -> KFD node id -> UUID -> render
minor**. The first two are partition-invariant; the rest are the fallback for consumer parts
that expose neither (e.g. gfx1030 has no OAM and one device per socket, so KFD/UUID suffice
because there are no partitions to confuse). Plugins report the raw truth (full BDF with
function, real `partition_index` from the PCIe function, `oam_id` when known); the core does the
grouping and the whole-GPU handle is authoritative for per-GPU keys. This replaces RDC's single
undocumented KFD-id `memcmp`.

Result: one canonical GPU ordinal per physical device, grouped by socket, with a partition list,
plus a per-plugin handle table so a read routes to the right plugin-local device (and, for a
partition the plugin does not model, to the whole-GPU handle).

## Plugin ABI

A plugin is a shared library exporting one C symbol,
`const gpum_plugin_v1* gpum_plugin_entry_v1(void)`, returning a static vtable with
`init`, `shutdown`, `enumerate`, `list_metrics`, and a batched `read`. Everything is plain C so
plugins can be written in C, C++, or Rust.

Design choices:
- **Query-then-route**: plugins advertise metrics; the core builds a `key -> plugin` routing
  map. When two plugins offer the same key a configurable provider priority resolves it (default
  `amdsmi` before `rocprofiler`), not RDC's silent last-writer-wins.
- **Batch read** so a plugin can coalesce (rocprofiler samples many counters in one pass; amdsmi
  reads one `gpu_metrics` struct for many fields).
- Loaded with `dlopen(RTLD_NOW | RTLD_GLOBAL)`. `RTLD_GLOBAL` is required so rocprofiler-sdk
  discovers the plugin's `rocprofiler_configure` symbol at `hsa_init`.
- **Crash isolation**: a plugin is probed in a forked child (dlopen + init + enumerate) before
  being loaded for real. Some backends `LOG(FATAL)`/abort on unsupported configs that cannot be
  caught in-process (rocprofiler-sdk 1.1.0 aborts when a 64-agent CPX layout exceeds its node-id
  encoding); if the child aborts, the parent skips the plugin instead of dying.

## Layers

```
cli/                 gpumetrics             CLI (discover / list-metrics / read / dmon)
include/             gpumetrics/*.h         public C API, C++ header, plugin ABI
src/                 libgpumetrics          core: registry, correlation, plugin host, collector
plugins/amdsmi       libgpumetrics_amdsmi.so
plugins/rocprofiler  libgpumetrics_rocprofiler.so
tests/               gtest unit + hw integration + amd-smi comparison
rust/                FFI (-sys) + safe wrapper
```

## Robustness rules (lessons from RDC)
- No sentinel-value overloading in the public API: "unsupported" is `GPUM_ERR_UNSUPPORTED`,
  never a magic `0`/`UINT64_MAX`. Plugins translate backend sentinels (0xFFFF, UINT32_MAX,
  UINT64_MAX) at the boundary.
- Enumerate topology once and cache handles; never re-enumerate per read.
- rocprofiler is process-global (single HSA, `RTLD_GLOBAL`, unsets `HSA_TOOLS_LIB`); treat it as
  a non-unloadable plugin with a lifetime owned by the core.
- A plugin that aborts on init must not take down the tool (fork-probe isolation).
