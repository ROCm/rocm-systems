<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocddi

rocddi is a private Rust device interface for ROCm runtime frontends. It
models passive topology endpoints, explicit activation, memory, and resource
ownership. GPU execution, PCI attachment, and operating-system handles enter
through separate capabilities and platform modules.

This crate belongs to the four-package `runtimes` Cargo workspace. The runtime
components are early-access and outside the repository default installation.
The `rocddi` crate is an `rlib`. It installs no headers, exports no C symbols,
and promises no stable Rust ABI.

The only platform implementation is the Linux KFD/DRM GPU driver on x86-64
and AArch64. Public topology and memory records keep Linux and GPU fields in
kind- or platform-specific modules. The private `Driver` trait covers
discovery, activation, context policy, metadata allocation, and shutdown. Its
associated device-state type retains each activated endpoint's dependencies.
`GpuDriver` is the complete GPU backend interface. It requires allocation and
virtual-memory operations, creates both direct and kernel-mediated queues, and
provides GPU presentation, cache control, timing, traps, and stream monitoring.
`AllocationOperations` and `VirtualMemoryOperations` describe mechanisms that
can also be implemented by a non-GPU driver. A created queue implements either
`UserQueueResource` or `KernelQueueResource`; these resource contracts define
distinct progress and teardown rules but do not represent additional drivers.
`AddressSpaceInfo` and `CachedInfo` expose the immutable values needed by
generic resource owners.
Host-only storage uses the selected host OS services. CPU cache maintenance is
qualified by `cpu_cache.rs`. Neither acquires a GPU driver context.

A public `Session` is the caller's logical scope for endpoint identity,
activation, driver context lifetime, and shutdown. It can discover and activate
several devices; construction activates none. The AMDF and HSA frontends each
use a `Session` as their rocddi root. A session owns a set of independently
identified driver instances under one requested context lifetime. The
current Linux constructor installs a KFD driver instance. Each activated
device retains only its owning driver instance and state. Session activation
rejects an endpoint from another driver instance, and shutdown requires every
device and resource owner to release its driver instance first. Clones share
the driver set.
A session may share KFD files and VMs according to the chosen policy.
`DriverContextLifetime::Process` permits a shared process-owned KFD context
and retained VM bindings. `DriverContextLifetime::Session` requires
the driver to release its context and bindings at session shutdown.

The private `DriverInstance` enum stores shared references to concrete driver
implementations. `DeviceDriverState` pairs one such reference with the state
acquired for a single activated endpoint. One driver instance can activate
several devices. `Endpoint::driver_instance` carries the numeric identity that
routes a passive snapshot to its owning driver instance.

The driver registry routes endpoint activation by the instance identity in
each passive record. With multiple drivers, enumeration stages their records
and rejects duplicate endpoint IDs before delivering any record. Opening an ID
first finds its owning driver and rejects a collision. A driver discovery
failure leaves the caller with no partial enumeration. Shutdown checks every
driver instance for live owners before closing any driver; a failed close can be
retried without repeating drivers that already closed successfully. Activated
devices from different drivers never share an address domain.

The common driver and generic allocation owners are exercised with a fake CPU
driver in tests. Those tests cover discovery, activation, foreign endpoints,
allocation, and failed cleanup without KFD, DRM, or PCI. Linux host-storage
tests cover both context lifetime policies and retryable unmapping. The
concrete `Session` tests check routing across two non-GPU drivers, endpoint ID
collisions, and coordinated shutdown retry. A fake direct-queue resource and a
detached virtual-memory driver check foreign-session rejection, mapping
occupancy, cleanup retry, and the order in which native owners and drivers are
dropped.

The common `Driver` trait imposes no GPU operations on non-GPU drivers. A new
implementation adds a `DriverInstance` and `DeviceDriverState` variant and is
installed by the session constructor. A GPU implementation satisfies
`GpuDriver`; another device family implements only the shared resource
contracts whose semantics it provides. Generic resource owners retain their
concrete driver and cleanup state. Public allocation, virtual-memory, and
queue types contain private concrete-owner variants, so their Rust signatures
do not contain KFD types. Linux descriptor, IPC, SVM, and event-page interop
requires a KFD resource and checks that ownership before using KFD or DRM.
Both C frontends route Linux interop through platform modules.

## Architecture

The peer frontends and shared package are:

- `ddi/libamdf`, which implements the AMDF v5 table ABI and builds
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

- `session.rs` owns the public session, routes endpoint identities across its
  installed drivers, selects the installed driver for a multi-device virtual
  address reservation, and coordinates shutdown. Each driver retains the
  selected context lifetime policy;
- `topology/` owns passive endpoint metadata. `EndpointKind` separates CPU,
  GPU, NPU, and future endpoint kinds; PCI attachment is optional, while
  `topology::platform::linux` carries KFD and DRM identities and procfs/sysfs
  host facts needed by Linux compatibility frontends;
- `device.rs` owns explicitly activated endpoint state and kind-neutral
  introspection. `device/event.rs` owns driver-independent subscriptions and
  event records. `device/gpu.rs` defines the checked `GpuDevice` view; its
  `device/gpu/` children own copy, queue, profiling, and GPU event services.
  Cloned devices share family-specific service state, including the GPU copy
  pool. Linux KFD signal events live under `device/gpu/event/linux.rs` and are
  exposed through `device::gpu::event::linux`;
- `memory.rs` and `memory/` own driver-generic allocation,
  address-reservation, and mapping owners. A virtual-address reservation
  verifies that every device uses the selected driver, intersects their
  address apertures, and allocates mapping records before asking that driver
  to reserve the native range. `memory::interop::linux` contains DMA-BUF,
  KFD IPC, KFD SVM, and AIS file-transfer contracts used with Linux APIs and
  other processes;
- `driver.rs` defines the private driver and resource interfaces;
  `driver/instance.rs` owns installed-driver and activated-device routing, while
  `driver/resources.rs` pairs concrete drivers with allocation, virtual-memory,
  and queue owners. Validation and retryable cleanup stay in the generic owner
  modules. The Linux KFD and DRM implementation lives in
  `driver/linux_kfd.rs` and `driver/linux_kfd/`. Its `operations.rs` implements
  the shared driver contracts; `interop.rs` provides KFD and DRM sharing
  operations. KFD and DRM owners remain with that driver;
- `os.rs` selects host allocation and page-size discovery.
  Owned files use `std::fs::File` directly. `os/linux.rs` and `os/linux/` own
  Linux host services used by the KFD driver. `file.rs` owns raw descriptor
  transfer, positioned I/O, and DMA-BUF file identity. `memory.rs` owns page
  discovery, host allocation, and the fork marker page; `memory/reservation.rs`
  owns Linux virtual address reservations and file mappings.
  `process_identity.rs` publishes and checks the marker to reject owners
  inherited across `fork`, and `os/linux.rs` translates I/O failures. Linux
  positioned file I/O uses `std::os::unix::fs::FileExt`. Another OS
  implementation supplies host allocation and page-size operations without
  exposing its native handle to the core.
  Host allocation reports a numeric extent and retains failed cleanup for
  retry; `memory.rs` constructs the public information record. The core
  host-memory API uses these services without activating a device.
  `topology::platform::linux` remains public endpoint metadata for runtime
  frontends; it does not own host services;
- `cpu_cache.rs` qualifies the host CPU cache-line recipe and runs cache
  maintenance without an OS handle. The memory API and GPU queue setup use
  this shared CPU service.

Each activated KFD process connection owns one hardware-exception event and
one memory-exception event. Device checks and `device::event` subscriptions use a
single serialized observer for both events. After copying a native payload, the
observer rearms its manual-reset event so later faults and resets can be
reported. The DDI keeps unread records in a bounded buffer and retains the
latest record of each exception kind for later subscribers. Each subscription
has an independent cursor. If a subscriber falls behind enough to fill the
buffer, native polling pauses until it reads or drops its backlog. A
subscription retains the native event owner until it is dropped; the session
releases that owner after all devices and subscriptions are gone. Hardware
exceptions that report lost
memory latch connection-wide `DeviceLost` for operations. Memory faults and
hardware exceptions without lost memory remain notifications. Polling never
invokes a frontend callback while holding the native observation lock.

The [safety boundary and resource state guide](docs/safety.md) records the
native reachability rules shared by the core and both adapters.
Owned and caller-owned host pages use `HostMappingPolicy` to select the GPU
mapping. In the primary KFD context, coarse USERPTR omits the coherent flag;
fine adds it, extended also adds extended coherency, and uncached adds the
uncached flag while retaining coherent access. The caller keeps registered
pages mapped until native teardown succeeds or process teardown resolves
uncertain ownership. In a secondary KFD context, borrowed pages use DRM GEM
USERPTR. Coarse and fine use its default mapping; uncached selects an
uncached VM mapping. On GFX1201, extended also uses the default mapping: that
GPU's KFD extended USERPTR and DRM USERPTR mappings have the same effective
page type. Other GPU targets reject extended registration in a secondary
context. Owned host pages use `System` backing there.

A topology endpoint is passive metadata. It is not an activated `Device` and
does not authorize native execution or memory operations. The Linux KFD driver
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
`GpuCopySequence::begin_on_sdma_engine` binds its command context to one
advertised logical SDMA engine; submission and completion waits use that same
engine. The platform driver maps the logical index to its submission target.
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
the same Linux interop driver. It reports a known copied-byte prefix and
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
The Linux KFD driver allocates GWS on the queue before exposing its transport.
KFD permits one GWS queue per process and device, so a frontend that serves
multiple cooperative callers must share that queue. A failed GWS allocation
destroys the unpublished queue before releasing its ring and control backing;
an uncertain cleanup retains those owners through process teardown.

The `ring_memory` field on `QueueParameters::Aql` and
`QueueParameters::SdmaByEngine` selects a system ring or a host-visible local
ring. The Linux KFD driver admits local placement on GFX1201 when public VRAM
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

Scripted KFD tests in `src/driver/linux_kfd/tests/queue.rs` inject
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
