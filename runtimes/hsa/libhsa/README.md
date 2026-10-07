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

`hsa_amd_queue_create` accepts compute and SDMA descriptors. Compute queues
use the rocddi AQL control layout; their public header aliases the native
control page. SDMA queues use a separate HSA header because native SDMA
read and write pointers occupy different offsets. Both queue types retain a
rocddi queue owner until destruction succeeds. A failed native teardown keeps
the public handle and backing available for retry, and final shutdown retains
an unresolved owner for process teardown.

An AMD queue descriptor can place either ring in host-visible local memory on
GFX1201 by setting `HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF`. rocddi owns the
uncached, executable VRAM allocation and its CPU mapping. The queue descriptor
remains in system memory; `HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR`
is rejected. Ordinary `hsa_queue_create` calls allocate system rings for new
queues. HSA signal doorbell stores use rocddi's queue notification helper to
drain write-combined packet stores before notifying the GPU. Callers publish a
complete packet and advance the write index before ringing the doorbell.
Callers that write the doorbell mapping directly must provide the same
write-combined store ordering.

Cooperative queue creation uses one 16 KiB AQL queue per GPU agent. rocddi
allocates KFD GWS before the queue becomes visible. Repeated creates for that
agent return the same public queue and increase its reference count; only the
final matching destroy tears down the native queue. A create during final
teardown or after explicit inactivation returns out of resources and can be
retried after the final destroy. The shared queue has
no caller error callback or fixed scratch backing. AMD queue descriptors can
adjust its priority and CU mask. A local-ring descriptor can create the shared
queue in local memory. Later cooperative requests must use the same ring
placement because the existing queue cannot change its backing.

SDMA descriptors select a native engine ID or round-robin selection through
rocddi. Their public ring size and monotonic indices count bytes. The ring is
host-visible, while the read and write pointer info queries return addresses
mapped for a device producer. The public index functions use rocddi's host
pointer mappings directly; aligned handle layouts distinguish SDMA, AQL,
counted, and soft queues without taking the process registry lock. A caller
must serialize SDMA packet production, check ring space, and publish the
write pointer and doorbell after writing complete packets. CU masks and AQL
profiling control do not apply to SDMA queues. KFD SDMA queues have no
wavefront scheduling priority or queue-error callback event source. SDMA
descriptors requesting either option are rejected.

For compute queues, the doorbell info query returns the rocddi transport's
host mapping. A queue error event marks a queue that has faulted in virtual
memory. The process fault event supplies its address and reason, including when
the queue error arrives later. System event callbacks run after a bounded wait
for queue identification. Handlers can query per-queue fault status, address,
and reason; memory faults do not invoke queue error callbacks when a system
event handler is registered. Without one, the queue error callback receives
the fault status.

On Linux, the workspace-root shared package builds
`libhsa-runtime64.so.1` with `ROCR_1` default versions on its public HSA
symbols and supplies the conventional HSA library alias. Binary
compatibility still requires ABI and workload qualification.
Linux builds require an LLD linker to combine Rust's export map with the
`ROCR_1` symbol versions, including on the declared Rust 1.85 minimum version.
The CMake build stages this ABI with AMDF in one shared image. The AMDF
and HSA aliases share one rocddi native process context.

The AMD extension reports version 1.33. For a GPU agent,
`HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE` returns the KFD topology
limit and `HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE` returns the last
successful request in this runtime generation. `hsa_amd_agent_set_attribute`
validates the requested `size_t` against that limit, then calls rocddi to set
the VM's persisting L2 cache reservation. A per-agent mutex serializes native
updates and protects the cached request; a native failure leaves that value
unchanged. The call retains an in-flight device reference so final shutdown
waits for the ioctl to finish.

Image and sampler support is disabled on every GPU. The image extension is not
advertised, and its entry points return `HSA_STATUS_ERROR_NOT_SUPPORTED` while
retaining their public symbols.

PC sampling is unavailable on every GPU. The frontend retains its public
symbols for ABI compatibility, but system and agent extension queries do not
advertise the capability and session creation returns
`HSA_STATUS_ERROR_NOT_SUPPORTED`. Neither the frontend nor rocddi contains a
PC sampling execution path.

The standalone AMD profiler and AQLprofile extensions are not advertised.
AMD dispatch and asynchronous copy timing entry points remain available in
the AMD extension without an extension table.

The product name comes from qualified KFD topology text, with a generic AMD
name when that text is not a product name. ASIC family comes from the bound
DRM render node via rocddi's raw ioctl path,
with the topology value as a fallback. The HSA library does not require libdrm
at load time. CPU identity, memory capacity, and cache records come from
rocddi's Linux host facts; this frontend maps them to HSA agents and caches.
GPU cache-line size and VRAM bus width and maximum clock come from rocddi's
KFD cache and memory-bank topology. When a native memory-bank property is
absent, its query returns zero. The GPU cache-line query falls back to 256
bytes when KFD reports no L2 cache-line size.
The maximum scratch aperture and expert scheduling attributes use rocddi's
GPU geometry and activated KFD capability. The nonclustered kernel grid bound
matches the GFX1201 dispatch limit.
The GPU timestamp frequency comes from the bound DRM render node and is used
when translating GPU ticks to the system clock domain. If that frequency is
unavailable, tick translation reports an error instead of using a guessed
rate.
Linux host facts, driver identities, descriptors, IPC, SVM, and event imports
enter through this frontend's `platform/` adapter. Only a Linux adapter is
implemented.

Linear asynchronous copy and explicit copy-engine selection use rocddi's
bounded SDMA submission path on GFX1201. `hsa_amd_agent_preload` prepares
rocddi's idle default and advertised ring copy contexts unless the caller
skips blit preloading. Clock conversion takes live correlated counter samples
and needs no separate agent clock resource. The engine mask reports DRM DMA
rings within the GPU's advertised SDMA engine count and the 16 public engine
IDs. An explicit one-hot choice selects that ring. The preferred engine query
returns the lowest available ring. For a GPU pair, the source GPU owns the
copy, and both ranges must be accessible to it. The force-SDMA argument does
not change this path because its native copies use SDMA. Pitched rectangular
copy submits its rows through one native queue. Linear and broadcast batch
operations reuse one queue for their device and staged host entries in
descriptor order. Multi-entry copies from a CPU source use the first
destination GPU; peer destinations must be accessible to that GPU. The
frontend waits for dependency signals and retains signal storage and
runtime-owned memory until native retirement. It decrements the completion
signal after each successful copy. Each batch operation may use
its own completion signal or share one with other operations.
Each operation releases its memory borrow and decrements its signal when it
retires. An asynchronous failure sets the completion signal negative.
Uncertain native retirement retains the command buffer and all runtime-owned
operands for process teardown. When asynchronous copy profiling is enabled,
each accepted copy snapshots that setting. rocddi captures GPU ticks around
the first and last SDMA copy packets for linear, rectangular, and batch
copies; the query translates those ticks using the copy GPU's correlated
clock sample. CPU copies sample the same system clock before and after the
host copy. The completion signal retains the most recent completed copy's
clock domain and timestamps until another copy accepts it or the signal is
destroyed. Timing is published before the completion signal changes. Native
swap is unavailable on GFX1201.
Indirect copies and raw wait and signal operations are unsupported.
Zero-byte single copies and rectangles with a zero dimension return success
without changing the completion signal.
External semaphore imports and queue operations return unsupported on Linux,
matching the current KFD contract. Fabric virtual-memory handles are also
unavailable; their native export and import path requires UALink support.

Direct GPU copies require both operands to be mapped to the selected GPU.
GPU memory-pool allocations use private VRAM when the CPU-visible aperture
does not cover the device's local memory. These allocations return a GPU
address and report no host address or host-access flag in pointer info. CPU
agent access to that pool is unavailable. When the full local memory is
CPU-visible, pool allocations retain a host mapping. The AMD uncached pool
flag requests coherent and uncached native VRAM placement for pool
allocations and virtual-memory handles. For CPU pools, it requests coherent,
uncached rocddi-owned host pages. The primary KFD context maps ordinary CPU
pool pages with the pool's coarse, fine, or extended coherency policy. CPU
pool allocations in a secondary KFD context use rocddi system backing, which
is already uncached. Host locks in that context cannot request extended
coherency. Pointer info reports effective fine-grained and kernarg
flags for uncached pool allocations and host locks. The AMD PCIe flag requests
coherent access. Pointer info reports fine-grained global flags for those
allocations. The owning pool still governs peer access.
For linear asynchronous CPU-to-GPU and GPU-to-CPU copies, rocddi stages a
CPU-accessible operand that is not mapped to that GPU. The HSA worker reads
the source after dependency signals complete, copies a staged destination
only after native retirement, and then updates the completion signal. A
loaded executable source is snapshotted under the runtime registry lock; a
loaded destination is checked again before its staged bytes are written.
Ordinary host allocations can also be mapped with `hsa_amd_memory_lock` or
allocated through an HSA pool before SDMA submission. `hsa_memory_register`
is a placement hint and does not establish a GPU mapping. CPU-to-CPU copies
use a host worker with the same dependency and completion behavior. That worker
resolves loaded executable addresses under the runtime lock after dependencies
complete. Virtual memory copy ranges must fit within one mapped subrange and
have the required read or write access for the selected copy agent. Unmapping
or changing access waits until an accepted copy releases its borrow.

The logging ABI writes to stderr for a null stream and uses C stdio for a
non-null `FILE*`. The stream is borrowed from the caller. Each write holds the
logging lock, so replacing the stream, disabling logging, and shutdown wait for
earlier writes before returning. Callers keep an enabled stream open until an
explicit disable or replacement completes, or until final HSA shutdown. A
nonfinal `hsa_shut_down` only releases one runtime reference and keeps the
logging configuration. Logging calls made recursively from a custom C stream
callback are rejected.
Linux descriptor calls for memory and loader operations go through the
rocddi provider. The AIS file read and write entry points accept a borrowed
descriptor and a range within a GPU-owned allocation. CPU-visible ranges use
rocddi's positioned host file transfer, including partial-read and write
progress. GPU-only KFD VRAM uses the mapped allocation handle and rocddi's
native AIS ioctl. Each call transfers at most Linux `MAX_RW_COUNT` bytes and
leaves the descriptor's shared file position unchanged. The HSA registry lock
keeps the allocation live for the synchronous operation. On native ioctl
failure, the operation status receives the negative Linux errno when supplied
and `-EIO` otherwise. The copied byte count remains untouched because KFD's
input and output fields overlap.

Loaded executable segments retain a CPU mapping alongside their GPU address.
For executable loading, the supported GPU target is GFX1201. The loader also
accepts version 6 `gfx12-generic` objects with a generic target version when
loading for GFX1201. It rejects target feature settings that GFX1201 does not
support before allocating the load image. Program code objects use system
backing associated with a GFX1201 device.

File readers snapshot the code-object bytes and resolve the descriptor's path
while it is open. Loaded objects retain an encoded file URI independent of the
descriptor's later lifetime. Readers without a reopenable path use a memory
URI for their snapshot.

The loader applies dynamic AMDGPU `ABS32_LO`, `ABS32_HI`, `ABS32`, `ABS64`, and
`RELATIVE64` relocations within the checked load image before publishing its
symbols. Section-targeted static relocations are already applied in the code
object and are skipped. Undefined and common relocation symbols resolve against
executable definitions, preferring program definitions over those for the load
agent. Missing definitions return `HSA_STATUS_ERROR_VARIABLE_UNDEFINED`.
Malformed section or symbol references return
`HSA_STATUS_ERROR_INVALID_CODE_OBJECT`. Program linkage declarations must
have a definition before the code object is loaded, including when the object
has no relocation for that declaration.

`hsa_memory_copy` resolves an executable variable address through that owned
mapping and checks the complete requested range before copying. The runtime
registry lock keeps the loaded object live for the synchronous operation.
Host writes from synchronous copy, fill, and host-worker copy are fenced before
the entry point returns or its completion signal changes.
Synchronous copy uses host writes when both ranges are CPU-visible and rocddi
SDMA when a range is GPU-only. rocddi stages an ordinary host range in owned
system memory when the selected GPU cannot directly access it. Asynchronous
SDMA copies also handle GPU-only ranges. Synchronous fill uses host writes for
CPU-visible HSA allocations and loaded variables, and rocddi SDMA fill for
GPU-only ranges. Synchronous native
calls retain their allocations through retirement without holding the runtime
registry lock during the native wait.

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
