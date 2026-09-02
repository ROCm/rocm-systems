# Proxy-trace profiler plugin

This shared library implements the former built-in **ProxyTrace** feature as an
**NCCL/RCCL profiler v6 plugin** (`ncclProfiler_v6`; descriptor includes RCCL proxy-trace fields). RCCL records proxy diagnostics through the
profiler API (`ncclProfileProxyDiag`); this plugin stores the same maps and
dump format as the legacy implementation.

## Build

```bash
cd plugins/profiler/proxytrace
make
```

Requires a C++17 compiler and ROCm (`ROCM_PATH`, default `/opt/rocm`). The
standalone Makefile does not depend on a configured RCCL CMake build: plugin
ABI headers live in `nccl/` next to the source, and `proxy_trace.cc` is
compiled from this directory. Minimal extra types live in
`proxytrace_plugin_shim.h`.

Output: `librccl-profiler-proxytrace.so`

## Usage

**Option A — explicit plugin path**

```bash
export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-proxytrace.so
```

**Option B — RCCL auto-load (same directory as librccl)**

```bash
export RCCL_ENABLE_PROXY_TRACE=1
# RCCL tries to dlopen dirname(librccl)/librccl-profiler-proxytrace.so when
# NCCL_PROFILER_PLUGIN is unset.
```

**Option C — short name (requires `librccl-profiler-proxytrace.so` on `LD_LIBRARY_PATH`)**

```bash
export NCCL_PROFILER_PLUGIN=proxytrace
```

## Dump

On communicator destroy, RCCL calls `ncclProfilerProxyTraceDump` if exported by
the loaded profiler library. `ncclCommDump` uses the same hook.

The dump goes through NCCL's debug logger at `WARN` level, one line per proxy
operation, so **`NCCL_DEBUG` must be set** (`WARN` or more verbose) or the output is
discarded. Proxy operations only exist for inter-node traffic, so a single-node run
has nothing to report.
