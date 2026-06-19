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

Requires `libfmt` development headers (`fmt/format.h`) and a C++17 compiler. The
plugin does not use `plugins/profiler/example/`; minimal NCCL plugin types live
in `proxytrace_plugin_shim.h` next to the source.

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

**Option C — short name (requires `libnccl-profiler-proxytrace.so` on `LD_LIBRARY_PATH`)**

```bash
export NCCL_PROFILER_PLUGIN=proxytrace
```

Copy or symlink `librccl-profiler-proxytrace.so` to `libnccl-profiler-proxytrace.so`
if you rely on this naming convention.

## Dump

On communicator destroy, RCCL calls `ncclProfilerProxyTraceDump` if exported by
the loaded profiler library. `ncclCommDump` uses the same hook.
