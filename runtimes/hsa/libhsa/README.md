<!-- SPDX-License-Identifier: MIT -->

# libhsa frontend

This crate implements an HSA runtime ABI frontend over the private `rocddi`
Rust core. It is a peer of `libamdf`; it does not adapt through AMDF types or
tables.

## Current implementation

The frontend owns HSA initialization and shutdown, public handles, agents,
queues, signals, memory pools and regions, executable loading, profiling
state, callbacks, and status translation. rocddi supplies native discovery,
KFD activation, memory, queue, event, and cleanup mechanisms.

One process-global registry owns the active runtime and its reference count.
Final shutdown removes the runtime from that registry before stopping workers
and releasing native state. Blocking native work and user callbacks must remain
outside global registry locks.

On Linux, the workspace-root shared package builds
`libhsa-runtime64.so.1` with `ROCR_1` default versions on its public HSA
symbols and supplies the conventional HSA library alias. Binary
compatibility still requires ABI and workload qualification.
Linux builds require an LLD linker to combine Rust's export map with the
`ROCR_1` symbol versions, including on the declared Rust 1.85 minimum version.
The CMake build stages this ABI with AMDF in one shared image. The AMDF
and HSA aliases share one rocddi native process context.

Image and sampler support is disabled on every GPU. The image extension is not
advertised, and its entry points return `HSA_STATUS_ERROR_NOT_SUPPORTED` while
retaining their public symbols.

PC sampling is unavailable on every GPU. The frontend retains its public
symbols for ABI compatibility, but system and agent extension queries do not
advertise the capability and session creation returns
`HSA_STATUS_ERROR_NOT_SUPPORTED`. Neither the frontend nor rocddi contains a
PC sampling execution path.

The product name comes from qualified KFD topology text, with a generic AMD
name when that text is not a product name. ASIC family comes from the bound
DRM render node via rocddi's raw ioctl path,
with the topology value as a fallback. The HSA library does not require libdrm
at load time. CPU identity, memory capacity, and cache records come from
rocddi's Linux host facts; this frontend maps them to HSA agents and caches.
Linux host facts, driver identities, descriptors, IPC, SVM, and event imports
enter through this frontend's `platform/` adapter. Only a Linux adapter is
implemented.

All four asynchronous memory copy entry points return
`HSA_STATUS_ERROR_NOT_SUPPORTED`; no queued copy engine is implemented. Engine
status reports a zero availability mask, and asynchronous copy profiling cannot
be enabled. The exported symbols remain available for ABI compatibility.

The logging ABI accepts a null stream and writes to stderr using Rust's
standard library. Non-null C `FILE*` streams return
`HSA_STATUS_ERROR_NOT_SUPPORTED`; this frontend does not import C stdio.
Linux descriptor calls for memory and loader operations go through the
rocddi provider.

## Support and qualification matrix

| Surface | Current behavior | Current evidence |
| --- | --- | --- |
| Initialization, agents, and topology | Linux KFD/DRM GPU and CPU records | Unit and C ABI checks; GFX1201 combined-frontend activation in both orders |
| Signals and AQL queues | GPU queues with capability checks | Unit checks; GFX1201 callback and kernel dispatch probes |
| Memory pools, SVM, and IPC | Implemented entry points with capability checks | Unit and ABI checks; GFX1201 CPU kernarg pool allocation and access in the kernel comparison; no broad differential HSA memory workload |
| Executable loading | ELF parsing and code-object loading | Parser mutation tests and a GFX1201 kernel dispatch compared with installed ROCr; no general kernel workload qualification |
| SPM and clocks | Raw-pointer and native-call contracts | API and ownership tests; no hardware-wide profiling qualification |
| Async copy, images, and PC sampling | Explicitly unsupported | ABI entry points and rejection probes |

These checks establish only the named paths. A release needs functional
workloads, failure recovery, and concurrency checks on each supported GPU and
driver tuple. The GFX1201 probes and kernel comparison do not qualify
the remaining HSA execution surface.

## Build and test

From `runtimes`:

```sh
cargo test --package libhsa --locked
cmake -S . -B /tmp/rocddi-cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocddi-cmake
```

For the opt-in GFX1201 callback and capability checks, run
`tests/gpu/run_gpu_smokes.sh /tmp/rocddi-cmake/lib` and
`tests/gpu/run_kernel_comparison.sh /tmp/rocddi-cmake/lib/libhsa_runtime64.so`;
see [their README](tests/gpu/README.md) for requirements and scope.

## Qualification boundary

The implementation has broad unit coverage, but no production compatibility
claim follows from that coverage. A release still requires a pinned ROCr/header
baseline, independent C/Rust ABI checks, real workload and hardware
qualification, differential tests, and tooling interoperability. The shared
package permits both ABI frontends to activate the same primary GPU process
context.
