# rocjitsu Plugins

Execution plugins that hook into rocjitsu's simulation model. Each plugin
implements the `ExecutionPlugin` interface and receives callbacks for
dispatch lifecycle, workgroup/wavefront events, instruction execution,
memory instructions, register reads, and barriers.

## Plugin output

Plugins write diagnostic output through a configurable sink system rather
than directly to stderr. This makes output testable and redirectable.

| Variable | Default | Description |
|---|---|---|
| `RJ_SINKS` | `stderr` | Comma-separated list of sink types: `stderr`, `stdout`, `file` |
| `RJ_SINK_DIR` | *(none)* | Directory for file sinks. Required when `file` is in `RJ_SINKS` |

When `file` is in `RJ_SINKS`, each plugin writes to
`<RJ_SINK_DIR>/<plugin_name>.log`.

## Adding a new plugin

1. Implement `ExecutionPlugin` in a new subdirectory under `plugins/`.
2. Add the source to `CMakeLists.txt`.
3. Wire plugin creation in the interposer, gated by an environment variable.
4. Use `sink().write()` for all output.

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

## Architecture

The `ExecutionPlugin` interface (`execution_plugin.h`) defines hooks
that the compute unit and command processor call during execution.
Multiple plugins can be active simultaneously via `ExecutionPluginGroup`.
A static empty group is used as the default so the plugin group pointer
is never null and the no-plugin path has near-zero overhead.
