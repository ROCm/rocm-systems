---
myst:
    html_meta:
        "description": "Execution plugin system in rocJITsu, covering the ExecutionPlugin interface, multi-plugin dispatch, sink configuration, and built-in plugins for race detection and kernel logging."
        "keywords": "rocJITsu, ROCm, execution plugins, race detection, kernel logging, sink system, RJ_RACE, RJ_LOG, RJ_SINKS, RJ_SINK_DIR"
---
# Execution plugin system

rocJITsu provides a pluggable execution plugin system that lets analysis
tools observe simulation events without modifying the core hardware
model. Each plugin receives callbacks at well-defined points during
wavefront execution, and all output flows through a configurable sink
system that separates diagnostic content from its destination.

## Hook points

Plugins implement a common interface that defines several callback hook
points. The compute unit and command processor invoke these hooks during
simulation, and every active plugin receives each callback:

### Dispatch

Called when the command processor finishes processing a kernel dispatch
packet. The callback receives dispatch metadata including the entry PC,
grid dimensions, workgroup dimensions, register counts, and the kernel
name when symbol information is available in the code object.

### Memory

Called when a wavefront executes a memory instruction (global loads,
global stores, LDS reads, LDS writes). The callback provides the
instruction, address, and access width, giving plugins enough
information to track data flow and detect hazards.

### Register

Called when a wavefront reads a register (SGPR or VGPR). Plugins that
track register-level hazards use this callback to check whether a
pending memory operation has been properly waited on before the
destination register is consumed.

### Barrier

Called when a wavefront executes an `s_barrier` instruction. Plugins
that track inter-wave synchronization within a workgroup use this to
update epoch or ordering state.

### Waitcnt

Called when a wavefront executes `s_waitcnt` or one of its split-counter
variants. This lets plugins retire in-flight memory events that are now
known to have completed from the issuing wave's perspective.

## Multi-plugin dispatch

Multiple plugins can be active simultaneously. An aggregation layer
dispatches each callback to every registered plugin in sequence. When a
plugin is added to this group, the group assigns it a sink for output.
If no sink is explicitly configured, the default is standard error.

## Sink system

Plugins write all diagnostic output --- race reports, kernel logs,
profiling data --- through a sink abstraction rather than writing
directly to `stderr` or `stdout`. This makes output testable,
redirectable, and composable.

Three sink types are available:

| Sink type | Behavior |
|-----------|----------|
| `stderr` | Writes to standard error (the default). |
| `stdout` | Writes to standard output. |
| `file` | Writes to a per-plugin log file inside a specified directory. Each plugin writes to `<RJ_SINK_DIR>/<plugin_name>.log`. |

The `RJ_SINKS` environment variable accepts a comma-separated list so
that multiple sinks can be active at the same time. For example,
`stderr,file` sends output both to the terminal and to a log file.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RJ_RACE` | unset | Set to `1` to load the race detection plugin. |
| `RJ_LOG` | unset | Set to `1` to load the kernel logging plugin. |
| `RJ_SINKS` | `stderr` | Comma-separated list of sink types: `stderr`, `stdout`, `file`. |
| `RJ_SINK_DIR` | *(none)* | Directory for file sinks. Required when `file` appears in `RJ_SINKS`. |


For a full reference of all rocJITsu environment variables, see
[Environment variable reference](../reference/environment-variables.md).

## Built-in plugins

### Race detector

The race detector plugin hooks memory instructions, register reads,
barriers, and `s_waitcnt` events to detect data races within a
workgroup. It tracks in-flight memory events through a lifecycle ---
active, wave-complete, and retired --- and reports when a register or
LDS byte is accessed before the operation that produced it has been
properly synchronized.

The detector identifies three categories of hazard:

-   **VGPR races**: a vector register is read before a pending global or
    LDS load has completed.
-   **SGPR races**: a scalar register is read before a pending scalar
    load has completed.
-   **LDS races**: an LDS byte is read or written by one wave while
    another wave has an outstanding write to the same byte, without an
    intervening `s_barrier`.

Detection operates at byte granularity: D16 (half-register) loads flag
races only on the affected bytes, and LDS races are tracked per byte.

Enable the race detector and direct output to files:

``` bash
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output \
  rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./my_app
# Race reports are written to /tmp/output/race.log
```

The plugin name for file sinks is `race`. For a hands-on walkthrough,
see [Detect a missing barrier with the race detector](../tutorials/race-detection-walkthrough.md).

### Kernel logging

The kernel logging plugin records kernel dispatch metadata and detects
MMA (matrix multiply-accumulate) instruction usage:

-   **Kernel dispatches**: entry PC, grid dimensions, workgroup
    dimensions, register counts, and kernel name (when available from
    the code object).
-   **MMA detection**: reports the first MFMA or WMMA instruction seen
    in each dispatch.

The plugin name for file sinks is `logging`.

``` bash
RJ_LOG=1 rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./my_app
```

## Usage examples

Interactive use with default stderr output:

``` bash
RJ_RACE=1 rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./my_app
```

Save race reports to files for CI or scripted workflows:

``` bash
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output \
  rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./my_app
```

Send output to both stderr and a file simultaneously:

``` bash
RJ_RACE=1 RJ_SINKS=stderr,file RJ_SINK_DIR=/tmp/output \
  rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./my_app
```

## Related pages

-   [Detect a missing barrier with the race detector](../tutorials/race-detection-walkthrough.md) --- step-by-step race detection tutorial
-   [waitcheck hazard analyzer reference](../reference/waitcheck.md) --- static
    wait-hazard checker for AMDGPU code objects
-   [ConSan GPU LDS sanitizer reference](../reference/consan.md) --- DBI-based LDS
    data race sanitizer
-   [API reference: virtual machine](../reference/api-vm.md) --- virtual
    machine C API reference
