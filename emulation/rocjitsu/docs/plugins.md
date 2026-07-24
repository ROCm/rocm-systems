# rocjitsu Plugins

Execution plugins that hook into rocjitsu's simulation model. Each plugin
implements the `ExecutionPlugin` interface and receives callbacks for
wavefront dispatches, memory instructions, register reads, barriers, etc.

## Plugins

| Plugin | Location | Description |
|---|---|---|
| `RaceDetectorPlugin` | `race_detector/` | Hooks memory instructions, register reads, barriers, and `s_waitcnt` to detect data races. Reports violations with disassembly traces. See [race-detector.md](race-detector.md). |
| `KernelLoggingPlugin` | `logging/` | Logs kernel dispatches and detects MMA instruction usage. |

The race detector plugin contains both the core detection algorithm
(`race_detector/core/`) and the rocjitsu adapter (`race_detector/plugin.h`).

### Kernel Logging Plugin

The logging plugin records kernel dispatch metadata and detects MMA
(matrix multiply-accumulate) instruction usage:

- **Kernel dispatches**: entry PC, grid dimensions, workgroup dimensions,
  register counts, and kernel name (when available from the code object).
- **MMA detection**: reports the first MFMA or WMMA instruction seen in
  each dispatch.

## Enabling plugins

Plugins are loaded at runtime based on environment variables:

- `RJ_RACE=1` enables the race detection plugin.
- `RJ_LOG=1` enables the kernel logging plugin.

## Plugin output

Plugins write diagnostic output (race reports, profiling data, kernel
logs) through a configurable sink system rather than directly to stderr.
This makes output testable and redirectable.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RJ_SINKS` | `stderr` | Comma-separated list of sink types: `stderr`, `stdout`, `file` |
| `RJ_SINK_DIR` | *(none)* | Directory for file sinks. Required when `file` is in `RJ_SINKS` |

When `file` is in `RJ_SINKS`, each plugin writes to
`<RJ_SINK_DIR>/<plugin_name>.log`. Plugin names are fixed:
`race` for `RaceDetectorPlugin`, `logging` for `KernelLoggingPlugin`.

### Examples

```bash
# Interactive use (default) — output goes to stderr
RJ_RACE=1 LD_PRELOAD=librocjitsu.so ./my_app

# Save race reports to files (for test harnesses)
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output LD_PRELOAD=... ./my_app
# Race reports are in /tmp/output/race.log

# Both stderr and file simultaneously
RJ_RACE=1 RJ_SINKS=stderr,file RJ_SINK_DIR=/tmp/output LD_PRELOAD=... ./my_app
```

### Writing a plugin that uses sinks

Plugins inherit a sink from `ExecutionPlugin`. Use `sink().write(msg)`
for all output instead of `fprintf(stderr, ...)` or `std::cerr`:

```cpp
class MyPlugin : public ExecutionPlugin {
public:
  MyPlugin() : ExecutionPlugin("myplugin") {}

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    sink().write(std::format("[myplugin] dispatch {}\n", info.dispatch_id));
  }
};
```

The sink is assigned by the `ExecutionPluginGroup` when the plugin is
added. If no group configures a sink, the default is stderr.

## How it works

The `ExecutionPlugin` interface (`execution_plugin.h`) defines hooks
that the compute unit and command processor call during execution.
Multiple plugins can be active simultaneously via `ExecutionPluginGroup`.

### VGPR observation precision

`onAmdgpuWriteVgprLanes` observes instruction-level VGPR destinations rather
than VM/runtime storage writes. Memory-pipeline completion and internal
destination-preservation merges deliberately bypass the hook.

The current implementation does not provide precise write masks for DPP
instructions or for sub-dword SDWA destinations using `UNUSED_PRESERVE`:

- DPP execution may report EXEC lanes that are later preserved by row/bank or
  `BOUND_CTRL` masking.
- Partial-preserve SDWA execution may report a full-dword write even though
  unselected destination bytes are preserved.

DPP restoration and SDWA destination merging use raw storage, so they do not
emit additional synthetic callbacks. The remaining semantic callback is still
conservative. Read observation is also not precise for these encodings: DPP
source staging may report the full source wave, and partial SDWA source staging
may report broader lane or byte effects than the instruction architecturally
uses.

Plugins that require exact register hazards must classify DPP and partial SDWA
instructions from the before-execute callback and ignore their VGPR read and
write callbacks. The runtime does not suppress these callbacks automatically.
This gives unsupported instructions false-negative coverage rather than
allowing conservative callbacks to become false-positive diagnostics.
Ordinary, 64-bit, and packed 16-bit destinations remain supported.

Precise DPP/SDWA observation is deferred to an execution refactor that will
report architectural register effects directly instead of staging broad reads,
executing broad writes, and repairing preserved state afterward.


## Adding a new plugin

1. Implement `ExecutionPlugin` in a new `.cpp`/`.h` pair under `vm/plugins/`.
2. Add the source to `CMakeLists.txt`.
3. Register the plugin in `simulated_kfd.cpp` (gated by an environment variable).
4. Use `sink().write()` for all output — never write to stderr directly.
