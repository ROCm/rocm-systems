---
myst:
    html_meta:
        "description": "Runtime environment variables for rocjitsu path resolution and diagnostics."
        "keywords": "rocjitsu, ROCm, environment variables, runtime directory, diagnostics"
---
# Environment variable reference

rocjitsu reads environment variables at runtime for path resolution and
diagnostics. This page documents each variable's default value, allowed values,
and effect. Plugin selection uses JSON config entries or the rocjitsu CLI
`--plugin` flag; plugin sinks are configured with the JSON `sinks` object.

For plugin configuration details, see
[Execution plugin system](../conceptual/execution-plugins.md). For command-line
options that interact with these variables, see [rocjitsu CLI reference](rocjitsu-cli.md).

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
| **Effect** | Used to locate ROCm libraries and tools for optional ROCm-dependent tests when `BUILD_TESTING=ON`. An explicitly empty CMake `ROCM_PATH` disables environment and `/opt/rocm` fallbacks. It does not select the host compiler or install destination. |

### `ROCJITSU_RUNTIME_DIR`

Override for the runtime directory used by daemon socket resolution.

| | |
|---|---|
| **Default** | None (falls back to `XDG_RUNTIME_DIR`) |
| **Allowed values** | Any valid directory path |
| **Effect** | When set, the default daemon socket path resolves to `$ROCJITSU_RUNTIME_DIR/daemon.sock`. This takes priority over `XDG_RUNTIME_DIR`. `rocjitsu --attach default` and attach IDs use this runtime directory; explicit attach paths override it. |

### `XDG_RUNTIME_DIR`

Standard XDG base directory for runtime files.

| | |
|---|---|
| **Default** | System-dependent (typically set by the session manager) |
| **Allowed values** | Any valid directory path |
| **Effect** | When `ROCJITSU_RUNTIME_DIR` is not set, the daemon socket path resolves to `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock`. If neither `ROCJITSU_RUNTIME_DIR` nor `XDG_RUNTIME_DIR` is set, the fallback path is `/tmp/rocjitsu-<uid>/daemon.sock`. |
