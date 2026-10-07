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
  mapping owners. `memory::interop::linux` contains DMA-BUF, KFD IPC, KFD
  SVM, and AIS file-transfer contracts used with Linux APIs and other
  processes;
- `driver/` is the private downward-facing platform contract, with the current
  Linux KFD and DRM implementation under `driver/builtin/linux_kfd/`. Linux
  memory and event interop have separate driver contracts so future platform
  backends do not need to implement file-descriptor or KFD event operations.

The [safety boundary and resource state guide](docs/safety.md) records the
native reachability rules shared by the core and both adapters.
Caller-owned host registration uses coherent KFD USERPTR mappings. Requesting
uncached registration adds the native uncached flag while retaining coherent
GPU access. The caller keeps the complete host page cover mapped until native
teardown succeeds or process teardown resolves uncertain ownership.

A topology endpoint is passive metadata. It is not an activated `Device` and
does not authorize native execution or memory operations. The current backend
publishes only GPU endpoints, but GPU geometry and queue capabilities live in
the `Gpu` endpoint-kind payload instead of being mandatory universal fields.
Likewise, Linux identities and sharing mechanisms stay in Linux-specific
extensions rather than defining the core endpoint or memory contracts.
GPU topology also carries cache-line and VRAM memory-bank properties, including
bus width and maximum memory clock, so frontends can report native values
without duplicating sysfs parsing.
The Linux driver reads the GPU counter frequency from the render-node device
info and attaches it to correlated native clock samples.

GPU topology includes the native maximum persisting L2 cache reservation.
`GpuDevice::set_persisting_l2_cache_size` validates the request against that
limit and forwards it to the Linux driver. The driver submits
`DRM_IOCTL_AMDGPU_VM` on the render file bound to the activated KFD VM. rocddi
owns the native request and device lifetime; API frontends own any public
attribute values and status translation.

`GpuInfo::maximum_scratch_aperture_bytes` derives the scratch address-space
bound from the GPU generation and active XCC count. The activated
`GpuDevice::supports_expert_scheduling` view combines GFX12 capability with
the bound KFD interface version; KFD 1.20 or newer is required.

`GpuDevice::copy_linear`, `copy_rect`, `copy_rects`, and `fill_u32` provide
GFX1201 SDMA copies and dword fills through a bounded DRM submission context
and a rocddi-owned command allocation. `copy_rects` validates every shape
before submission and reuses one native queue across its entries. Clones of an
activated GPU share an idle copy context for the default path and each selected
DRM DMA ring. A completed copy or fill returns a healthy context to its slot;
concurrent operations acquire independent contexts.
`GpuDevice::preload_linear_copy` prepares the default context and every
advertised ring before the first copy. Failed or unretired contexts are never
reused.
`GpuCopySequence` uses the same queue and command allocation for an ordered mix
of device-to-device, host-to-device, and device-to-host linear copies. It stages
each host operand when that entry runs and retires each packet before starting
the next entry. Any operation failure makes the sequence terminal. rocddi
queries the DRM DMA IP ring mask for an activated GFX1201 device.
`GpuCopySequence::begin_on_sdma_ring` binds its command context to one
advertised DRM ring; submission and completion waits use that same ring.
The default copy path uses ring zero. The same native owner encodes OSS5 linear
and constant-fill packets with system cache control, splits large transfers
into bounded packets, and reuses command backing only
after native retirement. A failed wait that cannot prove retirement keeps the
queue and command allocation live and reports that operand backing must also
be retained. Frontends own dependency and completion signals, public pointer
validation, and operand lifetime. `copy_from_host` and `copy_to_host` use
rocddi-owned system staging, so the host slice is read before native work or
written after retirement. An uncertain copy keeps staging live for process
teardown. `supports_linear_copy` reports whether the native path is enabled
for the activated GPU.

Optional copy-sequence timing owns a system allocation for two GPU clock
values. After `enable_timing`, the first copy packet writes the start tick and
every copy packet writes the latest end tick using OSS5 global timestamp
commands in the same native submission as the copy. `finish_timing` reads the
raw ticks after final retirement. An uncertain submission retains the
timestamp allocation and command backing, and reports that the frontend must
retain its operands. Frontends convert the ticks into their public clock
domain.

`memory::interop::linux::ais_transfer` borrows a descriptor and a live mapped
VRAM allocation for one synchronous KFD AIS read or write. The allocation
owner checks the logical and native backing ranges and supplies its KFD handle;
the Linux call boundary submits at most `AIS_MAX_TRANSFER_BYTES` in one ioctl.
The operation uses a positioned file offset and does not change the shared
file position. A successful ioctl returns its copied-byte count and operation
status. On an ioctl error, the input and output fields overlap in the UAPI, so
the byte count is unknown and rocddi preserves the native errno without
replaying the operation. Linux storage and PCI P2P capability determine whether
KFD can execute a particular transfer.
For CPU-visible storage, `ais_host_transfer` uses positioned host I/O through
the same Linux interop provider. It reports a known copied-byte prefix and
errno on failure, including short reads at end of file. The frontend owns
public pointer validation and HSA status translation for both routes.

Direct SDMA queues use byte-addressed rings and monotonic 64-bit read and
write indices. `QueueParameters::Sdma` keeps the KFD-selected engine route.
`QueueParameters::SdmaByEngine` uses KFD queue type 4 and either a checked
engine ID or round-robin selection across the device's general and XGMI SDMA
engines. The round-robin counter belongs to the activated device VM. The
resolved ID is available in `QueueTransport::sdma_engine_id`; other queue
formats and the KFD-selected route return `None`. The queue owner retains its
ring, index page, and doorbell mapping through native teardown, including
ambiguous CREATE or DESTROY outcomes. Producers own packet encoding, ring
space, publication order, and completion waits.

An AQL request can set `global_work_sync` when the GPU advertises GWS. The
Linux backend allocates GWS on the native queue before exposing its transport.
KFD permits one GWS queue per process and device, so a frontend that serves
multiple cooperative callers must share that queue. A failed GWS allocation
destroys the unpublished queue before releasing its ring and control backing;
an uncertain cleanup retains those owners through process teardown.

The `ring_memory` field on `QueueParameters::Aql` and
`QueueParameters::SdmaByEngine` selects a system ring or a host-visible local
ring. The Linux backend admits local placement on GFX1201 when public VRAM
can hold the page-rounded ring. It maps that VRAM into the process, requests
uncached GPU pages with execute permission, and owns the mapping through
native queue teardown. Producers using the host mapping must publish complete
packets and advance the write index before calling `ring_doorbell`. That helper
drains write-combined CPU stores before the native doorbell write.

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

On a GFX1201 system with KFD and its DRM render node, run the opt-in native
Rust contract tests with:

```sh
cargo test -p rocddi --test cts --locked -- --ignored --skip gfx1201_ais_vram_file_contract
```

It copies and fills through ordinary allocations, copies through rocddi-owned
host staging and a device virtual-memory mapping, and validates copies on each
advertised DRM DMA ring. It checks profiled SDMA copy ticks against correlated
native GPU clock samples. It also submits a packet through a targeted user
SDMA ring with device-producer mappings, byte-index, and doorbell progress
checks. The AQL case submits a barrier through a compute queue and checks GPU
signal completion and queue retirement. The capability case checks expert
queue scheduling against KFD 1.20 or newer and the GFX1201 scratch aperture
against the reported XCC count. The GWS AQL case runs the same retirement
check with native GWS allocated before queue publication.
The local-ring AQL case runs the barrier through a host-visible VRAM ring and
requires a GPU with sufficient public VRAM. The GWS variant combines that ring
with cooperative synchronization. The local-ring SDMA case copies through a
host-visible VRAM ring and verifies native retirement.
The AIS case reads a file into private VRAM and writes it back. Run it with
`ROCDDI_CTS_AIS_DIR` set to a writable directory on P2P-capable block or NFS
storage:

```sh
ROCDDI_CTS_AIS_DIR=/path/to/p2p-storage \
  cargo test -p rocddi --test cts --locked gfx1201_ais_vram_file_contract -- --ignored
```

The CMake build from `runtimes/` stages the same combined shared image and
AMDF static archive. Native allocations, virtual mappings, and frontend pools
use the queried host page size, including on 64 KiB page hosts. GPU execution
on a 64 KiB AArch64 KFD host remains a separate qualification step.

The native C probes under `ddi/libamdf/tests/abi` and the GPU examples
under `ddi/libamdf/examples` can be built directly when those checks are
needed. GPU execution requires `/dev/kfd` and DRM render-node access.
