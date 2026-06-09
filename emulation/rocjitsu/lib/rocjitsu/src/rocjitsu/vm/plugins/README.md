# rocjitsu Plugins

Execution plugins that hook into rocjitsu's simulation model. Each plugin
implements the `ExecutionPlugin` interface and receives callbacks for
wavefront dispatches, memory instructions, register reads, barriers, etc.

## Plugins

| Plugin | Location | Description |
|---|---|---|
| `RaceDetectorPlugin` | `race_detector/` | Hooks memory instructions, register reads, barriers, and `s_waitcnt` to detect data races. Reports violations with disassembly traces. |
| `KernelLoggingPlugin` | `logging/` | Logs kernel dispatches and detects MFMA usage. |

The race detector plugin contains both the core detection algorithm
(`race_detector/core/`) and the rocjitsu adapter (`race_detector/plugin.h`).

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
RJ_RACE=1 LD_PRELOAD=librocjitsu_kmd.so ./my_app

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


## Adding a new plugin

1. Implement `ExecutionPlugin` in a new `.cpp`/`.h` pair in this directory.
2. Add the source to `CMakeLists.txt`.
3. Register the plugin in `simulated_driver.cpp` (gated by an environment variable).
4. Use `sink().write()` for all output — never write to stderr directly.
