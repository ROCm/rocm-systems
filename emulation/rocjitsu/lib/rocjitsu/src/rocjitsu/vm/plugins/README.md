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
    "logging": {}
  }
}
```

The bundled plugins are `race` (`RaceDetectorPlugin`) and `logging`
(`KernelLoggingPlugin`).

### Plugin ABI

Each plugin `.so` exports two `extern "C"` functions:

- `const PluginMetadata *rocjitsu_plugin_metadata()` — returns a pointer
  to static metadata: `abi` version, `name`, `contact`, `version`, and a
  `config_schema` JSON string.
- `std::unique_ptr<ExecutionPlugin> rocjitsu_plugin_create(const char *config_json)`
  — constructs the plugin from its resolved JSON configuration string.

Use the `ROCJITSU_DEFINE_PLUGIN` macro from `plugin_abi.h` to emit both
functions. The host validates the reported `abi` against the loader's
expected version before use.

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

With a config file `my_config.json` that contains a `plugins` section:

```json
{ "plugins": { "race": {} } }
```

```bash
# Interactive use (default) — output goes to stderr
rocjitsu --config my_config.json -- ./my_app

# Save race reports to files (for test harnesses)
RJ_SINKS=file RJ_SINK_DIR=/tmp/output rocjitsu --config my_config.json -- ./my_app
# Race reports are in /tmp/output/race.log

# Both stderr and file simultaneously
RJ_SINKS=stderr,file RJ_SINK_DIR=/tmp/output rocjitsu --config my_config.json -- ./my_app
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

1. Implement `ExecutionPlugin` in a new subdirectory. The plugin class
   must be constructible from `const char *config_json`.
2. Add a `plugin_export.cpp` that calls
   `ROCJITSU_DEFINE_PLUGIN(MyPlugin, "myname", contact, version, schema)`.
3. In `CMakeLists.txt`, add the object library and a
   `rj_add_plugin_so(myname <object_lib> <export_src>)` call so it builds
   `librocjitsu_plugin_myname.so`.
4. Use `sink().write()` for all output — never write to stderr directly.
5. Enable it by adding `"myname": { ... }` to the `plugins` section of
   the config file.
