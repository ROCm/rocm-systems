# libhsa frontend

> [!CAUTION]
> This frontend is part of the early-access rocddi runtime infrastructure. It is
> not a drop-in replacement for the production ROCr HSA runtime. Expect its
> implementation, packaging, deployment, and qualification details to move.

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

On Linux, Cargo builds `target/{profile}/libhsa_runtime64.so` with the ROCr
`libhsa-runtime64.so.1` SONAME and `ROCR_1` default versions on its public HSA
symbols. Cargo does not install the conventional SONAME symlinks. Binary
compatibility still requires ABI and workload qualification.
Linux builds require an LLD linker to combine Rust's export map with the
`ROCR_1` symbol versions, including on the declared Rust 1.85 minimum version.

Image and sampler support is disabled on every GPU. The image extension is not
advertised, and its entry points return `HSA_STATUS_ERROR_NOT_SUPPORTED` while
retaining their public symbols. The product name comes from qualified KFD
topology text, with a generic AMD name when that text is not a product name.
ASIC family comes from the bound DRM render node via rocddi's raw ioctl path,
with the topology value as a fallback. The HSA library does not require libdrm
at load time. CPU identity, memory capacity, and cache records come from
rocddi's Linux host facts; this frontend maps them to HSA agents and caches.

The logging ABI accepts a caller-owned C `FILE*`, but this frontend supports
only a null stream, which writes to stderr through Rust's standard library.
A non-null stream returns `HSA_STATUS_ERROR_NOT_SUPPORTED`. Linux descriptor
calls for memory and loader operations go through the rocddi provider.

## Build and test

From `runtimes/rocddi`:

```sh
cargo build --package libhsa --locked
cargo test --package libhsa --locked
```

## Qualification boundary

The implementation has broad unit coverage, but no production compatibility
claim follows from that coverage. A release still requires a pinned ROCr/header
baseline, independent C/Rust ABI checks, real workload and hardware
qualification, differential tests, and
tooling interoperability. Same-process use with `libamdf` is not supported while
the two shared libraries contain separate copies of process-global rocddi state.
