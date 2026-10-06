<!-- SPDX-License-Identifier: MIT -->

# rocddi

rocddi is a private Rust device interface for ROCm runtime frontends. It
models passive topology endpoints, explicit activation, memory, and resource
ownership. GPU execution, PCI attachment, and operating-system handles enter
through separate capabilities and platform modules.

This crate belongs to the four-package `runtimes` Cargo workspace. The runtime
components are early-access and outside the repository default installation. The `rocddi` crate is an `rlib`. It installs
no headers, exports no C symbols, and promises no stable Rust ABI.

The only native implementation is the Linux KFD/DRM GPU backend on x86-64
and AArch64. Public topology and memory records keep Linux and GPU fields in
kind- or platform-specific modules. Private driver traits separate provider,
host allocation, general allocation, virtual memory, GPU queue, and GPU
profiling services. The common provider session and host and device
allocation owners accept a fake CPU provider in tests. Those tests cover
discovery, activation, foreign endpoints, allocation, busy teardown, and
failed cleanup without KFD, DRM, or PCI. Fake GPU queue and detached
virtual-memory providers check foreign-session rejection, mapping occupancy,
cleanup retry, and the order in which native owners and providers are dropped.
The public `Session` selects the native backend for the build target. Both C
frontends route Linux interop through platform modules. Another OS needs its
own backend and frontend adapters.

## Architecture

The peer frontends and shared package are:

- `ddi/libamdf`, which implements the AMDF v3 table ABI and builds
  `libamdf.a`;
- `hsa/libhsa`, which implements early-access HSA and AMD HSA extension
  entry points;
- the workspace-root `rocddi-frontends` package, which links both entry
  point sets into one Linux shared object and packages it under AMDF and
  HSA library names. It is the workspace's only shared release artifact.

Build the shared runtime with CMake from `runtimes` as shown below. Its
`libamdf.so`, `libamdf.so.0`, and `libhsa_runtime64.so` aliases resolve to
the same `libhsa-runtime64.so.1` image and one rocddi process context. The
package retains its exact KFD and DRM file owners through process exit, so its
shared object uses `NODELETE`. The last active frontend
session disables KFD runtime enablement without closing the retained VM.
The AMDF static library remains a separate build with its own process context.
It is not a shared-process substitute for the shared image.

The frontends own public handles, statuses, callbacks, initialization and
shutdown, ABI validation, loaders, and tooling semantics. rocddi owns only
shared native mechanisms and their resource lifetimes. Neither frontend may
depend on the other.

The core source is organized by ownership domain:

- `session.rs` owns the root session lifetime and cross-device coordination;
- `topology/` owns passive endpoint metadata. `EndpointKind` separates CPU,
  GPU, NPU, and future endpoint kinds; PCI attachment is optional, while
  `topology::platform::linux` carries KFD and DRM identities and procfs/sysfs
  host facts needed by Linux compatibility frontends;
- `device.rs` owns explicitly activated endpoint state, core lifecycle checks,
  and kind-neutral introspection. `gpu/` is the checked GPU capability view and
  exposes GPU queues and profiling, with KFD events below `gpu::event::linux`;
- `memory/` owns provider-generic allocation, address-reservation, and
  mapping owners. `memory::interop::linux` contains DMA-BUF, KFD IPC, and KFD
  SVM contracts used to exchange backing with Linux APIs and other processes;
- `driver/` is the private downward-facing platform contract, with the current
  Linux KFD and DRM implementation under `driver/builtin/linux_kfd/`. Linux
  memory and event interop have separate driver contracts so future platform
  backends do not need to implement file-descriptor or KFD event operations.

The [safety boundary and resource state guide](docs/safety.md) records the
native reachability rules shared by the core and both adapters.

A topology endpoint is passive metadata. It is not an activated `Device` and
does not authorize native execution or memory operations. The current backend
publishes only GPU endpoints, but GPU geometry and queue capabilities live in
the `Gpu` endpoint-kind payload instead of being mandatory universal fields.
Likewise, Linux identities and sharing mechanisms stay in Linux-specific
extensions rather than defining the core endpoint or memory contracts.

### Ambiguous queue creation

Native queue creation borrows frontend-owned GPU addresses. HSA supplies an
inactive/error signal allocation and optional scratch allocation; AMDF may
borrow a public scratch memory object. The core owns the ring, pointer and EOP
backing, context storage, device VM, and KFD connection.

Before a successful native CREATE, an ordinary error releases acquired core
backing and the frontend drops its borrowed resources. If a later step fails
with a known queue ID, the core destroys that queue before releasing backing.
An EFAULT from CREATE can conceal a live queue ID, and a failed rollback
DESTROY may leave the queue live. In either case the core reports
`QueueBackingMayBeLive` for uncertain native ownership and retains its native
dependency graph without replaying an unknown or possibly recycled ID.
Both frontends pass external owners through `create_queue_with_dependencies`.
The core protects those owners before invoking native creation. On an uncertain
CREATE result or unwind, it retains them; on success it returns them with the
queue for the frontend to keep through destruction.
HSA supplies its signal event, signal storage, and scratch. AMDF supplies a
scratch memory child borrow that prevents public memory destruction. HSA's
installed KFD mailbox page has process lifetime. No public queue handle is
issued after failed creation. If a later unpublished rollback fails or
unwinds, `abandon_unpublished_with_dependencies` retains both the queue and
external owners. Uncertain resources remain retained until process teardown
unless a future recovery mechanism proves release.

Scripted KFD tests in `src/driver/builtin/linux_kfd/tests/queue.rs` inject
CREATE EFAULT and rollback DESTROY failures; a creation-closure panic covers
an unwind. The tests verify that uncertain KFD outcomes retain native backing
and the external owner, an unwind retains the external owner, certain failure
releases that owner, and success returns it to the caller. Both frontends use
these shared ownership operations. The
[GFX1201 C ABI fault probes](tests/gpu/README.md) exercise ambiguous CREATE
and failed rollback through both frontends.

Public API declarations live separately under `../../api-headers/include`. The
[runtime API headers README](../../api-headers/README.md) records their
authoritative sources and synchronization rules.

## Build and validation

Run all commands in this section from `runtimes`:

```sh
cargo build --workspace --locked
cargo build --workspace --release --locked
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
python3 ddi/libamdf/tests/abi/check_layout.py
python3 hsa/libhsa/tests/abi/check_layout.py
cmake -S . -B /tmp/rocddi-cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocddi-cmake
```

The CMake build from `runtimes/` stages the same combined shared image and
AMDF static archive. The CI workflow checks all workspace targets for
`aarch64-unknown-linux-gnu` with Rust 1.85 and runs the CPU-only suite on a
native AArch64 runner. These checks cover architecture-specific source and
host behavior. Native allocations, virtual mappings, and frontend pools use
the queried host page size, including on 64 KiB page hosts. GPU execution on
a 64 KiB AArch64 KFD host remains a separate qualification step.

The CI workflow also pins nightly 2026-09-25 for a focused Miri check of
callback-backed host storage. With that toolchain and its `miri` and
`rust-src` components installed, run:

```sh
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" \
  cargo +nightly-2026-09-25 miri test --package rocddi --locked host_storage::tests
```

CI also varies Miri's scheduler seed across eight runs of the concurrent
final-owner release test. This broadens the explored interleavings without
claiming exhaustive schedule coverage.

The native C probes under `ddi/libamdf/tests/abi` and the GPU examples
under `ddi/libamdf/examples` can be built directly when those checks are
needed. GPU execution requires `/dev/kfd` and DRM render-node access.
