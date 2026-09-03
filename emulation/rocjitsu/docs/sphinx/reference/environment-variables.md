---
myst:
    html_meta:
        "description": "Runtime environment variables for rocJITsu, including logging, race detection, output sinks, and path configuration."
        "keywords": "rocJITsu, ROCm, environment variables, RJ_RACE, RJ_LOG, RJ_SINKS, configuration"
---
# Environment variable reference

rocJITsu reads environment variables at runtime to control execution
plugins, diagnostic output, and path resolution. This page documents
each variable, its default value, allowed values, and effect.

For details on how plugins use these variables, see
[Execution plugin system](../conceptual/execution-plugins.md). For
command-line options that interact with these variables, see
[rocjitsu CLI reference](rocjitsu-cli.md).

## Execution plugins

### `RJ_RACE`

Enables the race detection plugin.

| | |
|---|---|
| **Default** | Disabled (unset) |
| **Allowed values** | `1` to enable |
| **Effect** | When set to `1`, the race detection plugin hooks memory instructions, register reads, barriers, and `s_waitcnt` to detect data races within a workgroup. Reports violations with disassembly traces. |

### `RJ_LOG`

Enables the kernel logging plugin.

| | |
|---|---|
| **Default** | Disabled (unset) |
| **Allowed values** | `1` to enable |
| **Effect** | When set to `1`, the kernel logging plugin records kernel dispatch metadata (entry PC, grid dimensions, workgroup dimensions, register counts, kernel name) and reports the first MFMA or WMMA instruction seen in each dispatch. |

## Output sinks

### `RJ_SINKS`

Selects where plugin diagnostic output is written.

| | |
|---|---|
| **Default** | `stderr` |
| **Allowed values** | Comma-separated list of: `stderr`, `stdout`, `file` |
| **Effect** | Controls the destination for plugin output such as race reports and kernel logs. Multiple sinks can be active simultaneously. When `file` is included, each plugin writes to `<RJ_SINK_DIR>/<plugin_name>.log`. Plugin names are fixed: `race` for the race detection plugin, `logging` for the kernel logging plugin. |

### `RJ_SINK_DIR`

Directory path for file-based sinks.

| | |
|---|---|
| **Default** | None |
| **Allowed values** | Any valid directory path |
| **Effect** | Required when `file` is included in `RJ_SINKS`. Plugin output files are written to this directory. For example, race reports are written to `<RJ_SINK_DIR>/race.log`. |

## Plugin log groups

### `RJ_LOG_GROUPS`

```{note}
The default value, allowed values, and precise effect of `RJ_LOG_GROUPS`
are not yet documented. Documenting this variable requires inspection of
the plugin logging implementation.
```

## Execution mode

### `RJ_FORCE_SCALAR`

```{note}
The default value, allowed values, and precise effect of
`RJ_FORCE_SCALAR` are not yet documented. Documenting this variable
requires inspection of the simulation engine's scalar execution path.
```

## Path resolution

### `ROCM_PATH`

Path to the AMD ROCm installation.

| | |
|---|---|
| **Default** | System-dependent |
| **Allowed values** | Any valid directory path pointing to a ROCm installation |
| **Effect** | Used to locate ROCm libraries and tools when rocJITsu needs to resolve ROCm components. |

### `ROCJITSU_RUNTIME_DIR`

Override for the runtime directory used by daemon socket resolution.

| | |
|---|---|
| **Default** | None (falls back to `XDG_RUNTIME_DIR`) |
| **Allowed values** | Any valid directory path |
| **Effect** | When set, the daemon socket path resolves to `$ROCJITSU_RUNTIME_DIR/daemon.sock`. This takes priority over `XDG_RUNTIME_DIR`. Used by the `--attach` CLI option to locate a running daemon. |

### `XDG_RUNTIME_DIR`

Standard XDG base directory for runtime files.

| | |
|---|---|
| **Default** | System-dependent (typically set by the session manager) |
| **Allowed values** | Any valid directory path |
| **Effect** | When `ROCJITSU_RUNTIME_DIR` is not set, the daemon socket path resolves to `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock`. If neither `ROCJITSU_RUNTIME_DIR` nor `XDG_RUNTIME_DIR` is set, the fallback path is `/tmp/rocjitsu-<uid>/daemon.sock`. |
