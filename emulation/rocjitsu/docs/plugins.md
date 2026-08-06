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

### Plugin ABI

The plugin boundary is a C-shaped ABI. Each plugin `.so` exports three
`extern "C"` functions:

- `const PluginMetadata *rocjitsu_plugin_metadata()` — returns a pointer
  to static metadata: `abi` version, `name`, `contact`, `version`, and a
  `config_schema` JSON string.
- `PluginHandle rocjitsu_plugin_create(const char *config_json)` —
  constructs the plugin from its resolved JSON configuration string and
  returns an opaque handle.
- `void rocjitsu_plugin_destroy(PluginHandle handle)` — destroys an
  instance previously returned by `rocjitsu_plugin_create`.

Allocation and deallocation stay on the plugin side of the boundary: the
host destroys each instance through the plugin's own
`rocjitsu_plugin_destroy` export. Use the `ROCJITSU_DEFINE_PLUGIN` macro
from `plugin_abi.h` to emit all three functions. The host validates the
reported `abi` against the loader's expected version before use.

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
`race` for `RaceDetectorPlugin`, `logging` for `KernelLoggingPlugin`.

### Profiled execution

Set the top-level `"profiled": true` key to wrap the plugins in a
profiled execution group, which emits per-hook timing data
(`HOOK_PROFILE` lines) to the configured sinks. With the default sink, timing
data goes to stderr; stdout sends it to stdout, and file sinks write it to
`<dir>/profile.log`.

Profiled execution requires the simulation engine to use `"num_threads": 1`.
Multithreaded configurations are rejected because the profiling counters are
not synchronized.

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
