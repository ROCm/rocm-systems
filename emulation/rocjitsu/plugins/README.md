# rocjitsu Plugins

Adapter layer that connects standalone plugin libraries (from
`emulation/plugins/` or elsewhere) to rocjitsu's execution model.
Each adapter implements the `ExecutionPlugin` interface and translates
rocjitsu events (wavefront dispatches, memory instructions, register reads)
into calls to the underlying plugin library.

> **Direction:** The long-term goal is for plugins to be developed and
> built entirely outside the rocjitsu tree, and loaded dynamically at
> runtime (e.g. via shared libraries). The current in-tree adapters are
> a stepping stone toward that model.

## Plugins

| Adapter | Core library | Description |
|---|---|---|
| `RaceDetectionPlugin` | `emulation/plugins/race-detector` | Hooks memory instructions, register reads, barriers, and `s_waitcnt` to feed the race detector. Reports violations with disassembly traces. |
| `KernelLoggingPlugin` | *(none)* | Logs kernel dispatches and detects MFMA usage. |

## How it works

Currently plugins are loaded at runtime based on environment variables:

- `RJ_RACE=1` enables the race detection plugin.
- `RJ_LOG=1` enables the kernel logging plugin.

The `ExecutionPlugin` interface (`execution_plugin.h`) defines hooks
that the compute unit and command processor call during execution.
Multiple plugins can be active simultaneously via `ExecutionPluginGroup`.

## Adding a new adapter

1. Implement `ExecutionPlugin` in a new `.cpp`/`.h` pair in this directory.
2. Add the source to `CMakeLists.txt`.
3. If it wraps a standalone library, add that library to `emulation/plugins/`
   and link it in `CMakeLists.txt`.
4. Register the plugin in `simulated_driver.cpp` (gated by an environment variable).
