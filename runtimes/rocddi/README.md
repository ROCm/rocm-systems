# rocddi

> [!CAUTION]
> This is early-access runtime infrastructure. Expect API, packaging, and
> deployment details to move while rocddi is integrated into ROCm Systems. It
> is not part of the repository's default build or installation.

rocddi is a private, implementation-neutral Rust device interface for ROCm
runtime frontends. Its universal layer models passive topology endpoints,
explicit activation, memory, and recoverable resource lifetimes without making
PCI attachment, GPU execution, or a particular operating-system handle model
mandatory. Kind-specific capabilities and platform interoperability are
available through explicit child modules.

The directory is a self-contained three-package Cargo workspace. The `rocddi`
crate is an `rlib`, not a public C API or preload target. It installs no
headers, exports no C symbols, and does not promise a stable Rust ABI. The only
implementation currently provided is the Linux KFD/DRM GPU backend on x86-64
and AArch64. The neutral public contracts are intentionally shaped so future
CPU, GPU, and NPU backends, including Windows backends, do not need placeholder
KFD, DRM, file-descriptor, PCI, or GPU-only fields.

## Architecture

The current peer frontends are:

- `frontends/libamdf`, which implements the AMDF v3 table ABI and builds
  `libamdf.so` and `libamdf.a`;
- `frontends/libhsa`, which implements early-access HSA and AMD HSA extension entry
  points and builds the Cargo artifact `libhsa_runtime64.so`.

The frontends own public handles, statuses, callbacks, initialization and
shutdown, ABI validation, loaders, and tooling semantics. rocddi owns only
shared native mechanisms and their resource lifetimes. Neither frontend may
depend on the other.

The core source is organized by ownership domain:

- `session.rs` owns the root session lifetime and cross-device coordination;
- `topology/` owns passive endpoint metadata. `EndpointKind` separates CPU,
  GPU, NPU, and future endpoint kinds; PCI attachment is optional, while
  `topology::platform::linux` carries KFD and DRM identities needed by Linux
  compatibility frontends;
- `device.rs` owns explicitly activated endpoint state, core lifecycle checks,
  and kind-neutral introspection. `gpu/` is the checked GPU capability view and
  exposes GPU queues and profiling, with KFD events below `gpu::event::linux`;
- `memory/` owns the platform-neutral allocation, address-reservation, and
  mapping model. `memory::interop::linux` contains DMA-BUF, KFD IPC, and KFD SVM
  contracts used to exchange backing with Linux APIs and other processes;
- `driver/` is the private downward-facing platform contract, with the current
  Linux KFD and DRM implementation under `driver/builtin/linux_kfd/`. Linux
  memory and event interop have separate driver contracts so future platform
  backends do not need to implement file-descriptor or KFD event operations.

A topology endpoint is passive metadata. It is not an activated `Device` and
does not authorize native execution or memory operations. The current backend
publishes only GPU endpoints, but GPU geometry and queue capabilities live in
the `Gpu` endpoint-kind payload instead of being mandatory universal fields.
Likewise, Linux identities and sharing mechanisms stay in Linux-specific
extensions rather than defining the core endpoint or memory contracts.

Public API declarations live separately under `../api-headers/include`. The
[runtime API headers README](../api-headers/README.md) records their
authoritative sources and synchronization rules.

## Build and validation

Run all commands in this section from `runtimes/rocddi`:

```sh
cargo build --workspace --locked
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

The native C probes under `frontends/libamdf/tests/abi` and the GPU examples
under `frontends/libamdf/examples` can be built directly when those checks are
needed. GPU execution requires `/dev/kfd` and DRM render-node access.
