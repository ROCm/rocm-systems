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
queues, signals, memory pools and regions, executable loading, images, profiling
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
