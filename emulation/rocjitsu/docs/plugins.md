# rocjitsu Plugins

Execution plugins that hook into rocjitsu's simulation model. Each plugin
implements the `ExecutionPlugin` interface and receives callbacks for
wavefront dispatches, memory instructions, register reads, barriers, etc.

## Plugins

| Plugin | Location | Description |
|---|---|---|
| `RaceDetectorPlugin` | `race_detector/` | Hooks memory instructions, register reads, barriers, and `s_waitcnt` to detect data races. Reports violations with disassembly traces. See [race-detector.md](race-detector.md). |
| `KernelLoggingPlugin` | `logging/` | Logs kernel dispatches and detects MMA instruction usage. |
| `ThroughputPlugin` | `throughput/` | Reports per-dispatch and aggregate wave-instruction MIPS with an exclusive instruction-family breakdown. |
| `PerfsimPlugin` | `perfsim/` | Adapts gfx1250 execution observations to an external Perfsim backend implementing FFM observer APIs v8 through v13. Built only when explicitly enabled. See the [Perfsim adapter README](../lib/rocjitsu/src/rocjitsu/vm/plugins/perfsim/README.md). |

The race detector plugin contains both the core detection algorithm
(`race_detector/core/`) and the rocjitsu adapter (`race_detector/plugin.h`).

### Throughput Plugin

The throughput plugin counts one instruction whenever a wavefront reaches the
synchronous before-execute or asynchronous issue hook. Counts are **wave instructions**, not
active-lane operations. It reports one JSON object per completed dispatch and
one aggregate object at shutdown using the `rocjitsu.throughput.v2` JSONL
schema. Each object contains wall time, total wave instructions, MIPS, and an
exclusive breakdown into `scalar`, `vector`, `matrix`, `lds`, `global`,
`control`, and `other`; the family counts always sum to the total. Each family
reports `untimed_instructions` and `execution_timing_valid`. A family containing
untimed async work sets validity to `false` and emits null `execution_seconds`
and `execution_mips`; counts and `dispatch_mips` remain valid. Otherwise it
reports `execution_seconds` measured between this plugin's before- and
after-execute callbacks, `execution_mips` using only that family-local time, and
`dispatch_mips` using the complete dispatch time. Scheduler gaps, dispatch
setup, runtime work, and time spent executing other families are not included
in `execution_seconds`. Program terminators close their interval in the
wave-halt hook because they intentionally have no after-execute callback.
When other execution plugins are enabled, their before/after hooks can fall
inside this interval depending on registration order; run `throughput` alone
when comparing family timing. The timestamps themselves are outside the
measured interval but still add observer overhead to the run.

For a summary record, `wall_seconds` is the inclusive span from the earliest
dispatch begin to the latest dispatch end, including idle gaps between
dispatches. `dispatch_seconds_sum` is the sum of completed dispatch durations.
Dispatches that never reach the execution-end callback are omitted from both
per-dispatch output and the summary.

Memory instructions take precedence over their scalar or vector encoding. The
`lds` and `global` families describe the instruction's pre-routing pipeline tag
or mnemonic fallback: `lds` covers DS/local-memory instructions, while
`global` covers global, scalar-memory, flat, buffer, image, and scratch
instructions. A later shared-aperture FLAT-to-LDS remap is therefore still
reported as `global`. Their `execution_seconds` measure synchronous instruction
execution/address generation, not later routing, deferred pipeline completion,
or stalls charged to wait instructions. Matrix includes MFMA, SMFMAC, WMMA,
and SWMMAC instructions. Control covers branches, waits, barriers, termination,
no-ops, sleeps, and delays.

For a machine-readable report:

```json
{
  "plugins": { "throughput": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/rocjitsu-throughput" }
}
```

The report is written to `/tmp/rocjitsu-throughput/throughput.log`.

### Kernel Logging Plugin

The logging plugin records kernel dispatch metadata and detects MMA
(matrix multiply-accumulate) instruction usage:

- **Kernel dispatches**: entry PC, grid dimensions, workgroup dimensions,
  register counts, and kernel name (when available from the code object).
- **MMA detection**: reports the first MFMA or WMMA instruction seen in
  each dispatch.

### Perfsim compatibility plugin

The optional Linux-only Perfsim adapter loads a separately built Perfsim backend at
runtime. RocJITsu does not build, link, vendor, or install Perfsim and does not
write a persistent Perfsim trace; Perfsim continues to own its `GPUCSIM_*`
configuration and reports. See the [Perfsim adapter README](../lib/rocjitsu/src/rocjitsu/vm/plugins/perfsim/README.md)
for the runtime-first setup, independent build boundary, compatibility
requirements, limitations, and validation procedure.

## Enabling plugins

Plugins are compiled into standalone shared objects named
`librocjitsu_plugin_<name>.so` and discovered at runtime through the
standard dynamic-linker search path (`librocjitsu_plugin_*.so` are
installed next to the interposer, and the launcher adds that directory to
`LD_LIBRARY_PATH`).

A plugin is enabled by listing it in the `plugins` section of the
rocjitsu config file. The key is the plugin name (the `<name>` in
`librocjitsu_plugin_<name>.so`) and the value is a JSON object with the
plugin's configuration:

```json
{
  "plugins": {
    "race": {},
    "logging": {},
    "throughput": {}
  }
}
```

Plugin loading is best-effort by default: an unavailable or invalid entry is
logged and skipped. Set the optional top-level `"require_all_plugins": true`
when every configured plugin is required; startup then fails if any entry
cannot be loaded. With strict loading, `plugins` must be an object when present,
though omitting it is valid and creates an empty plugin group.

The default bundled plugins are `race` (`RaceDetectorPlugin`), `logging`
(`KernelLoggingPlugin`), and `throughput` (`ThroughputPlugin`). `perfsim`
(`PerfsimPlugin`) is available only in builds configured with
`ROCJITSU_ENABLE_PERFSIM_PLUGIN=ON`.

### Enabling plugins from the mirage CLI

When launching a workload through mirage, plugins can be selected on the
command line with `--plugin <name>` instead of editing a config file.
mirage injects each selected plugin into the rocjitsu config it synthesises
for the run (and, for containerised profiles, bind-mounts the plugin's
`.so` next to the interposer). The flag is repeatable and merges with any
plugins the profile already enables:

```bash
# Enable the race detector and the kernel logger for a single run.
mirage run --plugin race --plugin logging -- ./my_app

# Same, when starting a session.
mirage session start --profile mi350x --plugin race
```

Each `--plugin` enables the plugin with its schema defaults. Plugins that
take required arguments, or runs that need custom sink settings, are
configured through a profile or an explicit `--config <file>`.

### Plugin loader boundary

Plugins are repository-owned components built and shipped with rocJitsu. The
loader boundary does not provide compatibility or versioning for independently
built plugins; the host and plugins must always be rebuilt together.

Each plugin `.so` exports three `extern "C"` functions:

- `const PluginMetadata *rocjitsu_plugin_metadata()` — returns a pointer
  to static metadata: `name` and a `config_schema` JSON string.
- `PluginHandle rocjitsu_plugin_create(const char *config_json)` —
  constructs the plugin from its resolved JSON configuration string and
  returns an opaque handle.
- `void rocjitsu_plugin_destroy(PluginHandle handle)` — destroys an
  instance previously returned by `rocjitsu_plugin_create`.

Allocation and deallocation stay on the plugin side of the boundary: the
host destroys each instance through the plugin's own
`rocjitsu_plugin_destroy` export. Use the `ROCJITSU_DEFINE_PLUGIN` macro
from `plugin_exports.h` to emit all three functions. The host validates the
required exports.

### Config schema

The `config_schema` string describes the accepted config keys. Each key
maps to an object with a `type` (`string`, `number`, or `boolean`), an
optional `description`, and an optional `default`. Keys without a
`default` are required. Example:

```json
{
  "argname": { "type": "string", "description": "does something important", "default": "defaultvalue" },
  "requiredarg": { "type": "number" }
}
```

The loader merges defaults, validates types, checks for required keys,
and passes the resolved JSON object to `rocjitsu_plugin_create`.

## Plugin output

Plugins write reports and logs through a configurable sink system rather than
directly to stderr.
This makes output testable and redirectable.

### Sink configuration

Sinks are configured from an optional top-level `sinks` object in the
rocjitsu config (the same file that lists the `plugins`). There are no
sink-related environment variables.

| Key | Default | Description |
|---|---|---|
| `types` | `["stderr"]` | Array of sink types: `stderr`, `stdout`, `file` |
| `dir` | *(none)* | Directory for file sinks. Required when `file` is in `types` |

When `file` is in `types`, each plugin writes to
`<dir>/<plugin_name>.log`. Plugin names are fixed:
`race` for `RaceDetectorPlugin`, `logging` for `KernelLoggingPlugin`,
`throughput` for `ThroughputPlugin`, and `perfsim` for adapter diagnostics. Perfsim's
own report remains controlled by its `GPUCSIM_*` configuration.

### Examples

Interactive use — output goes to stderr (the default):

```json
{ "plugins": { "race": {} } }
```

```bash
rocjitsu --config my_config.json -- ./my_app
```

Save race reports to files (for test harnesses):

```json
{
  "plugins": { "race": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/output" }
}
```

```bash
rocjitsu --config my_config.json -- ./my_app
# Race reports are in /tmp/output/race.log
```

Send output to both stderr and a file simultaneously:

```json
{
  "plugins": { "race": {} },
  "sinks": { "types": ["stderr", "file"], "dir": "/tmp/output" }
}
```

> Note: plugins can also be selected on the mirage command line with
> `mirage run --plugin <name>` (see "Enabling plugins from the mirage
> CLI" above). Sink selection is still driven entirely by the config file
> shown here.

### Writing a plugin that uses sinks

Plugins inherit a sink from `ExecutionPlugin`. Use `sink().write(msg)`
for all output instead of `fprintf(stderr, ...)` or `std::cerr`:

```cpp
class MyPlugin : public ExecutionPlugin {
public:
  explicit MyPlugin(const char *config_json) : ExecutionPlugin("myplugin") {
    (void)config_json;
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    sink().write(std::format("[myplugin] dispatch {}\n", info.dispatch_id));
  }
};
```

The sink is assigned by the `ExecutionPluginGroup` when the plugin is
added. Writes through sinks assigned by one group are serialized at the
group's fanout boundary, including writes from asynchronous plugin workers.
If no group configures a sink, the default is stderr.

`KernelDispatchInfo` reports the effective LDS allocation in
`lds_size_bytes`, the descriptor-selected `wave_size`, the configured
execution `code_target`, and the workgroup-cluster dimensions. The code target
describes the simulator ISA selected for execution; it is not source-code-object
or translation provenance.

## How it works

The `ExecutionPlugin` interface (`execution_plugin.h`) defines hooks
that the compute unit and command processor call during execution.
Multiple plugins can be active simultaneously via `ExecutionPluginGroup`.

### VGPR observation precision

`onAmdgpuWriteVgprLanes` observes instruction-level VGPR destinations rather
than VM/runtime storage writes. Reads and writes carry the architectural lane
and byte masks for masked, sub-dword, and multi-register operands. Internal
storage operations used to preserve unaffected register state do not produce
additional architectural callbacks.

Asynchronous memory operations are modeled separately. The race detector
records their register dependencies when they are issued. A later completion
updates storage without emitting the same instruction-level write again.
Synchronization retires the corresponding outstanding operations using the
counter family captured at issue. This distinguishes legacy combined
`vmcnt`/`lgkmcnt` waits from split `loadcnt`, `storecnt`, `dscnt`, and `kmcnt`
waits on newer targets.

### Scalar register identity

`onAmdgpuReadScalarRegister` and `onAmdgpuWriteScalarRegister` carry a
`RegisterRef` whose class and index identify the architectural register. This
keeps plugins independent of encoded selector values and of the simulator's
physical storage layout. In particular, ordinary SGPRs and wave-private TTMPs
have distinct identities even though both can appear in scalar operand fields.

The older `onAmdgpuReadSgpr` hook continues to expose physical SGPR indices for
compatibility. New consumers that need architectural identity should use the
typed scalar-register hooks. Scalar writes have only the typed callback.

### Dispatch threading

Callback policy is derived from the plugins contained by an
`ExecutionPluginGroup`. An empty group returns before dispatch or locking. The
group divides hooks by frequency and synchronization cost:

- Lifecycle, dispatch, workgroup, wavefront, and barrier callbacks are
  infrequent. The group takes one recursive mutex before iterating its plugins,
  so two infrequent callbacks cannot overlap across simulation partitions. With
  the default hot-hook policy, an infrequent callback can still overlap a
  high-frequency callback. Recursive acquisition lets a callback synchronously
  read registers and fire register-observation hooks without deadlocking.
- Instruction before/after, memory-routing, tensor-DMA memory, and
  register-access callbacks are high-frequency and run concurrently with both
  other high-frequency callbacks and infrequent callbacks by default. Each
  callback is scoped to a wavefront below the simulation's shader-engine
  partition granularity. During the before-instruction callback, a memory
  instruction exposes its decoded wait-counter obligations and completion-order
  metadata through `amdgpu_memory_issue_info()`, before address or store-data
  operands are read. This metadata describes operations
  routed through the scalar, vector, and local memory pipelines; it is not a
  complete inventory of non-memory events, such as messages and timestamp
  queries, that hardware wait counters may also track.

A plugin whose high-frequency callbacks reach shared mutable state may override
`requires_serial_hot_hooks()` to return `true`. The group samples that stable
policy once when the plugin is added and then takes the same group mutex around
every high-frequency callback, serializing it with the infrequent callbacks
without a per-instruction scan of the plugin list. Plugins that protect their
own shared state should retain the parallel default.

### Observing memory accesses

There are two memory hooks, and they see different things.
`onAmdgpuRouteMemoryInstruction` fires before routing decides anything: the
instruction still carries the address space it decoded as and the addresses it
computed. `onAmdgpuMemoryAccessRouted` fires once routing has settled and
reports a `MemoryAccessObservation` describing the access the memory system is
actually about to be asked for — the pipeline it was issued to, the wait
counter it will post to, and the addresses it was issued with. The difference is
not cosmetic: a FLAT access into the shared aperture decodes as global and is
issued to the LDS pipeline with its addresses rewritten and its counter changed,
so an observer using the earlier hook charges it against the wrong cache at an
address the memory system never uses.

`decoded_space` preserves the instruction's original address-space family
independently of its effective `route`, so explicit SCRATCH remains distinct
from a FLAT access that resolves to scratch or LDS. When routing changes an
address, `pre_routing_addresses` contains the original per-lane span and
`addresses` contains the effective span. The former is empty when no address
changed; when present, both spans have `wavefront_size` entries and only lanes
in `valid_lane_mask` are meaningful.

Building the observation is real work on the per-instruction path, so it is
skipped entirely unless a contained plugin asks for it. A plugin that overrides
`onAmdgpuMemoryAccessRouted` must also override `observes_memory_routing()` to
return `true`; the group samples this policy when each plugin is added, and its
conservative default is `false`. Overriding the hook alone is silent — the
plugin simply never sees an access.

The observation's spans borrow execution-owned storage and are valid only for
the duration of the callback. A plugin that keeps one must copy them.

Tensor DMA transfers execute directly rather than through the ordinary memory
routing path. `onAmdgpuTensorDmaMemoryAccess` reports them separately after the
instruction and any descriptor-requested atomic-barrier arrival return normally.
Its `addresses` member is a callback-lifetime `TensorDmaAddressView` over the
in-bounds global element bases whose requests were attempted, in execution order
with duplicates preserved. Empty and fully masked transfers produce no callback.
Consumers must opt in through `observes_tensor_dma_memory_access()`. A consumer
that retains addresses must use `size()` to budget and allocate the event, then
call `copy_to()` during the callback; a failed copy means the observation is
malformed and must be rejected.

SGPR owner resolution is skipped when no contained plugin observes scalar
register reads. Plugins that consume neither `onAmdgpuReadScalarRegister` nor
`onAmdgpuReadSgpr` should override
`observes_sgpr_reads()` to return `false`. The group samples this policy when
each plugin is added. Its conservative default is `true`, so existing plugins
continue receiving SGPR read callbacks unless they explicitly opt out.

Pass the complete sink configuration to the group constructor and add plugins
before publishing the group to simulation components. `add()` is not
thread-safe, and the group must remain immutable while callbacks may dispatch
concurrently.

## Adding a new plugin

1. Implement `ExecutionPlugin` in a new subdirectory. The plugin class
   must be constructible from `const char *config_json`.
2. Add a `plugin_export.cpp` that calls
   `ROCJITSU_DEFINE_PLUGIN(MyPlugin, "myname", schema)`.
3. In `CMakeLists.txt`, add the object library and a
   `rj_add_plugin_so(myname <object_lib> <export_src>)` call so it builds
   `librocjitsu_plugin_myname.so`.
4. Use `sink().write()` for all output — never write to stderr directly.
5. Audit shared mutable state reached by high-frequency hooks against both
   high-frequency and infrequent callbacks. Override
   `requires_serial_hot_hooks()` when that state cannot be protected within the
   plugin.
6. Override `observes_sgpr_reads()` to return `false` when the plugin does not
   consume `onAmdgpuReadSgpr`, and `observes_memory_routing()` to return `true`
   when it does consume `onAmdgpuMemoryAccessRouted`. A consumer of
   `onAmdgpuTensorDmaMemoryAccess` must likewise return `true` from
   `observes_tensor_dma_memory_access()`.
7. Enable it by adding `"myname": { ... }` to the `plugins` section of
   the config file.
8. Return `true` from `supports_async_instructions()` only when the plugin accepts
   issuer-thread issue notifications and concurrent same-wave helper register
   hooks. Keep the default `false` for ordered event state or complete
   architectural snapshots. One non-opted-in plugin disables async execution
   for the entire group.

## Asynchronous arithmetic

Plugins default to synchronous execution. `supports_async_instructions()` opts
into `onAmdgpuAsyncInstructionIssued`, replacing the ordinary before/after
pair for offloaded instructions. The group samples this capability on `add()`;
all contained plugins must opt in.

The notification runs on the issuer after a helper accepts the instruction;
helper register hooks may already have run. Issue callbacks may read metadata,
not register values: even a source can alias a destination being written.
Callbacks must not retain instruction or wave references. There is no completion
notification. Holding the callback mutex does not provide a complete
architectural snapshot.

Throughput and kernel logging support this contract. ConSan keeps synchronous
execution until its dependency-event access and diagnostic context support
concurrent register hooks within one wave. Plugins and the host must be rebuilt
together, as for other execution-plugin interface changes.
