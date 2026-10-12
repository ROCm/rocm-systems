<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# AMDF API support and implementation map

This map describes the implementation in this tree. Hardware results are
recorded separately in the [validation record](../tests/README.md).

The [vendored headers](../../../api-headers/README.md) synchronized from
`hrx-system@bd24215e6a5d1e12570c892356fd5b343ac38a6d` define the sole public contract.
This page records implementation support; it does not define another ABI or
promise every service described by those headers. The libamdf frontend translates
the ABI to private rocddi Rust types and its private platform `Driver`.

## Negotiation and table composition

`amdf_query_api` selects core ABI v5 from the caller's inclusive version range.
The immutable table has 40 nonnull function slots. GPU extension v1 has six
slots; the XDNA extension is absent. Table presence describes library
composition, while endpoint capabilities, profiles, and queue families describe
which requests the provider can fulfill. Invalid inputs are validated even on
unsupported paths; callers must not expect every rejected request to have the
same status.

The tables are in [src/lib.rs](../src/lib.rs). Negotiation
performs no allocation, discovery, activation, or native call, and tables
remain valid until library unloading. No global initialization or symbol-by-
symbol compatibility adapter is required.

## Core slots

| Slots | Current behavior and limits | Implementation |
|---|---|---|
| `query_extension` | GPU v1 negotiated independently; unknown/XDNA extension unsupported. | [lib.rs](../src/lib.rs) |
| `instance_create`, `instance_destroy` | Explicit callback allocator and driver context lifetime policy; inert creation. Both PROCESS and INSTANCE requests work for CPU services. An inherited instance rejects work before touching callbacks, files, or locks. KFD shutdown disables only an owned runtime before closing its file; failed shutdown preserves retry ownership. | [instance.rs](../src/instance.rs) |
| `endpoint_enumerate`, `endpoint_open`, `endpoint_query_info`, `endpoint_close`, `endpoint_query_queue_family_info` | Passive Linux KFD discovery and cached GPU/family facts. Opening does not activate a device. Endpoint info includes the Linux DRM major/minor identity for inter-provider correlation. Family facts include the exact native queue encoding feature mask. Endpoint IDs describe discovery identities, not native file descriptors. | [instance.rs](../src/instance.rs) |
| `device_destroy` | Explicit teardown with queue-child BUSY checks. Memory dependencies are caller-enforced, as the header requires. | [instance.rs](../src/instance.rs) |
| `instance_enumerate_memory_scopes`, `device_enumerate_memory_scopes`, `memory_scope_query_info` | Instance SYSTEM scope and live-device enumeration of endpoint-owned LOCAL scopes where storage is reported. | [memory.rs](../src/memory.rs) |
| `memory_scope_query_device_profile` | CPU CREATE/REGISTER; single- and multi-device SYSTEM CREATE/REGISTER; SYSTEM EXPORT; one-native-VM SYSTEM IMPORT when DRM qualification is available; and LOCAL CREATE for the physical owner plus qualified direct peers. Registration requires the declared host cacheability. Multi-device requests require identical access requirements and intersect their address envelopes. Distinct-GPU IMPORT is unadvertised. A LOCAL request must include its owner; every peer needs a cached directional XGMI or kernel-validated PCIe route. | [memory.rs](../src/memory.rs) |
| `memory_create`, `memory_query_info`, `memory_query_access_info`, `memory_query_address`, `memory_destroy` | Construction establishes every requested access before publication. Multi-device SYSTEM and LOCAL allocation maps one backing through the ordered distinct KFD VM list at one common GPU VA and publishes one immutable access/address record per request. LOCAL backing is allocated on the scope owner even when it is not first in caller order. Repeated live consumers of one VM reuse its mapping. GPU registration maps the caller's page cover while preserving its logical host range. Registration of an exact provider-owned SYSTEM host view borrows its existing backing and VM access. Provider-created allocations report zero payload prefix and exact native rounding. Destruction consumes the public memory handle even when native release reports an error. | [memory.rs](../src/memory.rs) |
| `memory_map`, `host_mapping_query_info`, `host_mapping_cache_control`, `host_mapping_destroy` | Explicit ranged host views. Known WB/WC recipes require qualified host instructions. Registered pages use the cacheability declared by the selected profile; current CPU and GPU registration profiles require write-back pages. | [memory.rs](../src/memory.rs) |
| `memory_query_pair_info`, `memory_scope_query_pair_info` | Concrete and pre-construction host/host, host↔device, and device→device descriptions for coherent SYSTEM memory on qualified PM4, AQL, and SDMA families. Producer WRITE and consumer READ are required. A device pair composes a queue-executed GLOBAL release and GLOBAL acquire. Single-owner, host-visible WC LOCAL pairs are qualified for GFX1201 PM4, AQL, and SDMA; LOCAL peers, unavailable identity, and unknown recipes are unsupported. Different valid backing identities fail the precondition. | [memory.rs](../src/memory.rs) |
| `memory_import`, `memory_export` | Same-device SYSTEM/GTT DMA-BUF import uses KFD placement plus DRM GEM creation and same-device handle facts, requires coherent/uncached GPU storage and a write-back CPU view, and maps exact requested GPU permissions. Import retains its own descriptor and consumes the external value only after complete success. Page-aligned source offsets, cross-process transport, physical identity, re-export, and failure preservation are qualified. EXPORT remains available on supported one- and multi-device SYSTEM CREATE profiles. Foreign API, distinct-GPU and LOCAL imports are unadvertised. | [memory.rs](../src/memory.rs) |
| `external_memory_release` | Clears a move-owned external value before invoking its release callback exactly once. Exported DMA-BUF values close their owned descriptor. | [memory.rs](../src/memory.rs) |
| `user_queue_query_info`, `user_queue_map`, `user_queue_mapping_query_info`, `user_queue_mapping_destroy`, `user_queue_query_status`, `user_queue_wait_consumed`, `user_queue_destroy` | PM4/AQL/SDMA host mappings and opt-in same- or peer-device mappings for AQL/SDMA, cached metadata, exact family format-feature propagation, explicit status observation, caller-directed waits, and consuming teardown after the mutation-free BUSY preflight. A native cleanup failure retains unreleased backing without a retryable public handle. PM4 status expands its ring-relative native read pointer into the current monotonic producer window. Peer mappings require one instance, common VA coverage, and a KFD-accepted doorbell route. Queue consumption is not application completion. | [queue.rs](../src/queue.rs) |
| `kernel_queue_query_info`, `kernel_queue_query_status`, `kernel_queue_refresh_status`, `kernel_queue_wait`, `kernel_queue_destroy` | GFX1201 PM4 and SDMA queues retain immutable creation facts, cached retirement and terminal state, a nonblocking native refresh, and a bounded native wait. Querying status performs no native call. The requested pending count, or the default 4096, is reported as an admission bound; the queue's private DRM context admits one unretired command and makes one nonblocking progress check before rejecting a full slot. Destruction returns BUSY before mutation while work remains. Other results consume the handle, retaining unreleased native resources on cleanup failure. | [kernel_queue.rs](../src/kernel_queue.rs) |
| `kernel_queue_request_notification` | The queue reports no supported native event types. Descriptor validation is available, and valid requests return UNSUPPORTED. | [kernel_queue.rs](../src/kernel_queue.rs) |

## GPU extension slots

| Slots | Current behavior and limits | Implementation |
|---|---|---|
| `gpu.endpoint_query_info` | Immutable passive GPU target geometry. Library composition alone does not qualify native support. | [instance.rs](../src/instance.rs) |
| `gpu.device_create`, `gpu.device_query_info` | Explicit GPU activation with PROCESS or INSTANCE driver context lifetime. PROCESS selects the primary KFD context; INSTANCE selects an instance-owned secondary context on UAPI 1.19 or newer. Both advertise host registration: PROCESS binds through KFD USERPTR, while INSTANCE maps a DRM GEM USERPTR in its acquired render VM. Each distinct device retains its VM binding under serialized KFD runtime coordination; recreation reuses that binding. A shared endpoint observer advances live handles' reset epochs on asynchronously observed hardware exceptions; device operations and queues also advance the epoch when they first report terminal loss. The info query reads the epoch without a system call. | [instance.rs](../src/instance.rs) |
| `gpu.user_queue_create` | Fresh PM4, AQL, or SDMA queue storage and sidecars; per-queue doorbell slot in the retained native mapping. GFX1201 PM4 reports ACQUIRE_MEM GCR and uses a fixed 4 KiB host-only, single-producer, normal-priority KFD COMPUTE queue. AQL reports the baseline zero feature mask. GFX1201 SDMA reports GCR plus explicit-system FENCE and not the GFX12.5 memory-scope encoding. AQL supports single/multiple producers and low/normal/high KFD priorities; SDMA supports a single producer at normal priority. Requiring AQL/SDMA device production creates and maps a KFD doorbell BO at the retained CPU/GPU VA and reports queue-device addresses for the ring and indices. Later peer mappings attach the queue backing and doorbell at that same VA. Optional AQL scratch borrows one validated read/write device range through successful queue destruction. Existing GFX1201 host-produced workloads, including upstream PM4 CTS execution, are qualified; same-device device-produced AQL and SDMA execution is hardware-qualified in both lifetimes; peer mapping and peer-device production remain unqualified. | [queue.rs](../src/queue.rs) |
| `gpu.kernel_queue_create`, `gpu.kernel_queue_submit` | Linux x86-64/GFX1201 PM4 and SDMA families advertise kernel publication. Each queue owns a private DRM context and completion timeline. Submission accepts one dword-aligned executable device range for the queue's exact device and reset epoch, without reading or translating command bytes. Other targets remain unadvertised. | [kernel_queue.rs](../src/kernel_queue.rs) |

## Ownership and cost boundaries

Public parents are borrowed: instance → endpoint → device → queue → producer
mapping, and scope/devices → memory → host mapping. A kernel queue borrows its
device. Registration of a provider-owned host view borrows its source memory.
The caller retains all
parents and excludes concurrent destruction. Memory and queue use through raw
device addresses remains the caller's lifetime responsibility. Internal native
owners preserve only the dependencies needed to finish their own cleanup.

All native control enters the private [Driver](../../rocddi/src/driver.rs).
Cached endpoint, family, scope, device, memory, host-mapping, queue, mapping,
and address queries read retained state without allocation, locking, lazy
initialization, or ownership-counter updates. Pair queries compose only the two
supplied sites' retained facts. User-queue status is an explicit native
observation and may update atomic terminal/progress state; waiting uses the
caller's timeout. Allocation, activation, discovery, and creation are cold
operations that can call the OS.
KFD runtime activation is serialized and occurs during first device activation,
before later GPU allocation and queue work. Queue creation rechecks admission;
runtime disable precedes KFD close and is retryable after failure. Native
`EBUSY` or `EEXIST` admission leaves the runtime unowned and close does not
disable it. An ambiguous activation result blocks replay and requires cleanup.
The native backend rejects inherited work before touching its allocator
callbacks, filesystem state, or locks that may have been held across `fork`.
Callback allocation failure must leave no partially published object. There is
no global allocator selector, task scheduler, journal, pool cache, or allocation
registry for general allocations; the instance has a cold list only for
recognizing its own SYSTEM host views during registration. The HSA runtime is a
peer under `hsa/libhsa/`. HIP,
CUDA-like pool reuse, graphs, and recovery remain frontend or consumer policy.

## Qualification boundaries

The [validation record](../tests/README.md) identifies the GFX1201 binary,
hardware and driver used for recorded GPU results, with runnable checks and
unqualified cases. The implementation supports Linux x86-64 and AArch64 builds;
GPU execution has been checked on one x86-64 GFX1201 host.

The XDNA extension is absent. Kernel queues on other GPU targets,
foreign or LOCAL memory import, dynamic AQL scratch growth, and dependency
bearing AQL barriers remain unsupported. Peer-device queue production,
multiple GPU execution, live reset and unplug, AArch64 GPU execution, and
cross-runtime stress need hardware qualification. The AMDF and HSA shared
aliases use one image and share rocddi process state. The separately linked
AMDF static archive has its own state.
