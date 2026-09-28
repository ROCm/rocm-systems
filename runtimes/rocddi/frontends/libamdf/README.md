# libamdf

> [!CAUTION]
> This frontend is part of the early-access rocddi runtime infrastructure.
> Expect implementation, packaging, deployment, and platform-qualification
> details to move while rocddi is integrated into ROCm Systems.

libamdf implements the AMDF native C ABI in Rust. Its sole public contract is
the seven headers under `api-headers/include/amdf/`, synchronized from
`hrx-system/libamdf` at `4aa34130de44c45d68a48575cebfd0ff0610c461` with only
the approved AMD copyright and MIT license preamble substitution.
The build produces `libamdf.so` and `libamdf.a`; `amdf_query_api` is the sole
exported C entry point. `libamdf.so` is an independent preloadable API frontend
and does not require an HSA frontend to be installed or loaded.

AMDF separates passive endpoint discovery from explicit device activation.
An instance owns its native connections and callback allocator. Upper runtimes
own device selection, suballocation, queue pooling, packet construction,
graphs, synchronization policy, and recovery. Native GPU mechanisms are
provided by the owning [rocddi](../../README.md) layer; this frontend owns the AMDF
ABI and translates it to the implementation-neutral core interface.

## Implemented services

The provider implements CPU-only CREATE and REGISTER profiles, single- and
multi-device SYSTEM CREATE and REGISTER, SYSTEM DMA-BUF EXPORT, LOCAL
allocations with qualified direct peer access, PM4/AQL/SDMA user queues, and
GFX1201 PM4/SDMA kernel queues.
Public and private VRAM have separate profiles.
GPU memory supports exact READ, READ|WRITE, READ|EXECUTE, and
READ|WRITE|EXECUTE access. A multi-device request uses one permission/property
set and one common GPU virtual address; unsupported combinations fail before
allocation. LOCAL requests must include the physical owner and every peer must
have a cached directional XGMI or kernel-validated PCIe route to that owner.
Provider-owned host mapping recipes currently require x86 CLFLUSH/MFENCE support.
Registration requires callers to declare the cache class promised by the
selected profile. The current CPU and Linux KFD paths accept qualified
write-back pages. GPU registration establishes an independent GPU address for
the caller's page cover and reports the same coherent SYSTEM cache transitions
as owned GTT. Registration of a provider-owned SYSTEM host view reuses its
existing backing and GPU mapping while borrowing the source allocation.
Host-visible LOCAL profiles additionally require a qualified native cache mode;
current qualification covers the ordinary GFX10.1–12.0 discrete path. Provider-owned profiles
requiring an unavailable host recipe return UNSUPPORTED. Host page size is
queried at instance creation; native GPU/queue support remains separately
qualified by the backend.

Same-device SYSTEM DMA-BUF import is advertised only when the Linux backend
can verify KFD placement, DRM GEM creation, and same-device handle facts using
the required DRM ioctls. The qualified path accepts coherent or uncached GPU
storage with a write-back CPU view and exact requested GPU permissions. Drivers
without those capabilities leave IMPORT unadvertised; SYSTEM CREATE still
supports same-provider GTT export, including page-aligned logical source
offsets and cross-process descriptors. Foreign API buffers, distinct-GPU
attachments, and LOCAL transport remain unsupported. Dynamic scratch
management and device cache-pair recipes outside the narrow qualified GFX1201
SYSTEM-memory PM4/AQL/SDMA paths are unsupported. Fixed
caller-supplied scratch is qualified for the GFX1201 wave32 AQL path.
Same-device AQL and SDMA production passed GPU execution under both lifetimes;
peer-device production has source and unit-test qualification only.
The XDNA extension table is absent. Direct PM4 is qualified only on Linux
x86-64 GFX1201 with a fixed 4 KiB host-produced ring, one producer, and normal
priority. Its raw native read pointer wraps in the ring, while status and wait
operations expand it into AMDF's monotonic dword frontier. Presence of a
callable table slot does not imply that every service or request is supported.
The [API support map](docs/api-support.md)
accounts for every table slot and separates current behavior from qualification.

## Ownership and native boundary

The public object model remains AMDF's object model. rocddi retains only the
native ownership needed to implement those objects safely and does not import
AMDF declarations or public lifetime rules.

| Object | Ownership and lifetime |
|---|---|
| Instance | Owns the copied host allocator and native connections; construction performs no discovery or device activation. |
| Endpoint | Query-only object borrowing its instance; caches immutable endpoint and family facts. |
| Device | Explicitly activated execution/address domain borrowing its endpoint. |
| Memory scope | Borrowed storage metadata embedded in its instance or endpoint. |
| Memory | Owns allocation-specific native state; borrows its scope and requested devices. |
| Host mapping | Owns an explicit host view and borrows its memory. |
| User queue | Owns native transport state and borrows its device. |
| Kernel queue | Owns a private native submission context and completion timeline; borrows its device. |
| Queue mapping | Owns a producer view and borrows its user queue. |

Discovery returns fixed-stride endpoint summaries and opaque identities.
Opening an endpoint does not activate it or establish memory access. CPU-only
storage requires no accelerator endpoint or device handle.

Callers keep public parents alive for each documented child lifetime. The
instance has a cold list to recognize its own SYSTEM host views during
registration, but no general allocation registry. Queue and mapping counters
enforce AMDF's BUSY cases; they cannot prove retirement of work that references memory through
opaque addresses. Concrete native owners retain the dependencies needed to
finish their own cleanup without changing public borrowing rules.

Constructors validate inputs and reserve metadata before publishing a complete
output. Failure leaves outputs unchanged and creates no public cleanup
obligation. Enumeration's documented BUFFER_TOO_SMALL count/prefix protocol
is the exception. Metadata uses the instance's copied allocator callbacks;
backing uses operating-system or driver allocation mechanisms. The callbacks
and their user data remain valid through successful instance destruction.
There is no global allocator selector or native attachment cache.

Destruction accepts a live handle by value. Success consumes it; the caller
must discard its aliases. No caller pointer slot is cleared. Native teardown
failure preserves cleanup ownership and progress for retry, without promising
continued access to transport after partial cleanup. Destruction does not
implicitly wait for completion. Callers serialize destruction against other
uses according to the imported headers.

If ambiguous native cleanup still retains callback-backed metadata, instance
destruction fails and preserves the remaining native owners. The caller keeps
its allocator and library alive; no allocation registry or recovery service is
introduced to conceal that failure.

PROCESS native lifetime permits kernel-owned process state to survive instance
destruction. INSTANCE requires library-owned native state to be released with
the instance. On KFD UAPI 1.19 or newer, this provider selects an
instance-owned secondary KFD context and supports GPU activation under both
lifetimes. Secondary contexts do not support USERPTR registration.

On the KFD provider, first-device activation acquires and publishes the retained
VM owner, then enables the KFD runtime before publishing the public device.
Runtime activation is serialized across concurrent callers. Recreating that
same device or activating another device acquires or reuses its distinct VM
binding under the same serialization. Queue creation rechecks runtime admission
before acquiring queue backing. Shutdown closes every retained VM binding,
disables the runtime, and then closes KFD; a failed step retains the remaining
state for retry. The backend rejects inherited native work before touching its
own callbacks, files, or locks. Native `EBUSY` or `EEXIST` activation leaves a
foreign runtime unowned and is never followed by disable; an ambiguous
activation outcome blocks replay and requires cleanup. The current one-GPU
test host cannot execution-qualify the second-device path.

## Memory and queues

The native memory boundary consists of AMDF scopes, profiles, resources,
accesses, and host mappings. A scope describes a storage source. A profile
qualifies a complete construction combination for the supplied device-access
requirements. This separates storage placement from device permissions,
address kinds, cache properties, and mapping support.

The instance supplies SYSTEM scope metadata without activating an accelerator.
CPU CREATE owns operating-system backing; REGISTER borrows caller storage and
never frees it. A GPU REGISTER profile pins the complete page cover in the
activated VM while preserving the exact logical range and caller ownership.
Live devices enumerate endpoint-owned LOCAL scopes. Host-visible and private
VRAM use separate profiles because their available capacities and mapping
limits can differ.

Construction selects one profile and supplies the complete ordered access set.
The current provider supports zero devices for CPU storage, one GPU for LOCAL
storage, or multiple activated GPUs for SYSTEM CREATE and REGISTER when every access has identical requirements. Multi-device SYSTEM
construction intersects the live VM apertures, reserves one common address,
and maps the backing to the ordered distinct KFD VM list before publishing the
resource. Repeated live consumers of one VM reuse its native mapping while
retaining separate access records. Access is never installed lazily by an
address query.

Information separates logical byte length, native allocation length,
alignment, backing properties, and per-device access properties. Host
visibility does not establish device coherence. HOST_COHERENT does not prove
that device cache release/acquire operations are unnecessary.

Host access uses an explicit mapping with a range, permissions, cacheability,
and available cache recipes. UNKNOWN means no qualified recipe; it does not
mean that a transition is a no-op. Multiple views can refer to subranges of
the same backing. The caller must destroy mappings and prove that all device
uses have stopped before destroying memory; those are ABI preconditions rather
than library-tracked ownership.

The SYSTEM DMA-BUF export path creates independent close-on-exec file
descriptors and identifies backing with the descriptor's device/inode pair.
Import consumes the caller's external value only after the qualified native
attachment is fully published. Rejected imports preserve the external value
and output; failed exports likewise preserve caller outputs and ownership.

A resource owns its native allocation and the progress of its own cleanup.
AMDF memory destruction consumes the public handle even when native release
fails; the internal owner completes safe remaining cleanup or retains native
state for process teardown. Memory borrows its scope/devices and does not extend
their public lifetimes. The instance's host-view list does not retain backing,
record arbitrary allocations, or defer cleanup.

Host-to-host pairs on one resource report transitions only when both local
recipes are known. The Linux x86-64 GFX1201 PM4, AQL, and SDMA 7.0.1 paths also
describe directional host-to-device acquire and device-to-host release for
coherent SYSTEM memory. The producer must have WRITE access and the consumer
READ access. The GPU transition is a queue-executed global operation; a
write-back host view needs no additional cache operation, but publication and
completion ordering remain caller responsibilities. Two device sites on
coherent SYSTEM backing compose the producer family's GLOBAL release with the
consumer family's GLOBAL acquire. Single-owner, host-visible WC LOCAL memory
also has directional host/device and device/device recipes on the qualified
GFX1201 AQL and SDMA families. LOCAL peer pairs, other device-involving recipes, and
UNKNOWN host recipes return UNSUPPORTED. For different resources,
unavailable physical identity returns UNSUPPORTED; different valid identities
return FAILED_PRECONDITION. Equal addresses alone prove neither shared backing
nor visibility. Failures leave the output unchanged. Direct LOCAL/VRAM peer
mapping has no peer cache-pair recipe and has not been execution-qualified on
a multi-GPU host.

Each GPU user-queue creation acquires a fresh native queue, ring, index
storage, and required sidecars. Its doorbell address is a per-queue slot in
a retained native mapping shared by that device's queues. The queue borrows
its device; each explicit producer mapping borrows its queue. PM4, AQL, and
SDMA expose explicit host producer mappings when their native transport meets
the advertised format. Requiring `DEVICE_PRODUCER` for AQL or SDMA creation also maps the
ring, indices, and process doorbell slice into the queue device's VM. Mapping
the queue for another device in the same instance lazily establishes common-VA
ring and index mappings and asks KFD to validate that device's peer doorbell
route. Queue backing keeps successful peer VM dependencies through destruction;
the shared process doorbell keeps its peer mappings through owning-device
teardown. PM4 does not advertise device production.

On Linux x86-64/GFX1201, PM4 and SDMA families also offer kernel publication.
Each kernel queue owns one private DRM context, a completion timeline, and one
atomic submission slot. The caller supplies a single immutable command range
with execute access in the queue device's current address domain. The provider
does not inspect or translate its bytes. The caller keeps the command and its
indirect dependencies live until the accepted submission retires. Cached
status is syscall-free; an explicit wait proves retirement before storage
reuse. An uncertain native submission becomes a failed accepted submission so
its storage cannot be reused prematurely.

Family and creation information specify format, producer mode, priority,
ring bounds, and achieved capabilities. There is no native C batch-creation
API or implicit queue pool. AQL supports single and multiple producers; PM4
and SDMA support a single producer. AQL accepts low, normal, and high KFD
scheduling priorities; PM4 and SDMA accept normal priority. Priority does not establish an
execution dependency, completion order, fairness, or latency guarantee. The
caller supplies the selected format's atomic publication and doorbell protocol.
The selected family's exact encoding features propagate unchanged through
queue and mapping information. GFX1201 PM4 reports ACQUIRE_MEM GCR. SDMA 7.0.1
reports the five-dword GCR and explicit-system FENCE encodings; its packets do
not use the GFX12.5 memory-scope fields. AQL format v1 reports the baseline zero
feature mask.

Progress reports consumed and producer frontiers in format-defined units. PM4
status brackets the native ring-relative read pointer with stable producer
samples and expands it into the current monotonic producer window. For AQL
multiple-producer queues, the producer frontier can include reserved
slots that have not been release-published. A wait target must already have
been release-published by the caller. Consumption of ring storage does not
prove completion of every memory access initiated by its commands.
The caller retains application resources through their actual device-use
lifetime, independently of queue mapping lifetime.

Destruction returns BUSY before mutation while mappings remain or the
producer and consumer frontiers differ. Once native teardown begins, failure preserves cleanup
ownership but blocks further transport use. Retry does not replay released
native IDs; ambiguous outcomes retain their dependencies until process
teardown. There is no hidden completion wait.

## Build and validation

From `runtimes/rocddi`:

```sh
cargo build --workspace
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

The C ABI probes in `tests/abi` and the [GPU examples](examples/README.md) are
standalone sources for explicit local qualification. Native GPU execution
requires `/dev/kfd` and the selected DRM render device. The pinned upstream
CTS source is recorded in [tests/README.md](tests/README.md).

The header pin and licensing are recorded in the
[API header provenance](../../../api-headers/README.md). The checked-in Rust
bindings and C layout probe are snapshots of those headers. Header changes
require regenerating and comparing both ABI declarations and layout values.
The rocddi layer contains a private core and two peer API frontends:

- `frontends/libamdf` builds `libamdf.so` and `libamdf.a`. It owns C ABI
  validation, the negotiated tables, and public handle lifetimes.
- `frontends/libhsa` builds `libhsa_runtime64.so`. It owns the HSA symbol
  ABI, process runtime, public handles, queues, signals, loading, and tooling
  semantics.
- The workspace root supplies implementation-neutral native mechanisms through
  domain modules for sessions, topology, activated devices, memory, queues,
  events, and profiling. Its private `driver/` layer owns the platform contract
  and Linux KFD implementation, while `host_storage.rs` implements fallible
  callback-backed ownership.

Rust types are internal implementation details. Consumers use the AMDF C API.
The HSA peer consumes the core directly and does not depend on libamdf.

Validation results and their limits are recorded in [tests/README.md](tests/README.md).

## Next work

The upstream CTS now qualifies a direct-PM4 SYSTEM-memory copy and concurrent
device recreation in every PROCESS linkage mode. The
[GFX1201 examples](examples/README.md) qualify a SYSTEM-memory SDMA copy, a
private and host-visible LOCAL SDMA round trips, a registered-host SDMA copy,
a qualified DMA-BUF subrange import/re-export SDMA copy, a scratch-backed
AQL copy-add dispatch from SYSTEM and LOCAL memory, and GPU-produced AQL
and SDMA queue packets through the imported tables. The LOCAL workload checks placement, exact device access,
private-map rejection, public WC mapping, prospective and concrete directional
cache pairs, CPU and SDMA access to public VRAM, 64 round trips, ring reuse,
completion, results, queue status, and ordered cleanup under both lifetimes.

The registered-host workload pins subpage-offset caller buffers with exact GPU
permissions, preserves their host addresses, obtains independent GPU addresses,
checks page-cover and alignment metadata, and exercises directional cache pairs.
It completes 64 checked copies and eight ring wraps before destroying the AMDF
objects and proving that the caller still owns usable pages.

The current DMA-BUF workload exports a page-aligned subrange of shareable
SYSTEM memory, imports it with exact READ access through an independent
descriptor, and verifies backing identity and metadata. A child process
attaches the same backing through a fresh instance. The parent destroys the
source allocation, copies from the live import through SDMA, re-exports it,
and releases every descriptor. PROCESS and INSTANCE passed this workload on
the qualified GFX1201 host. Import remains unadvertised when the required DRM
ioctls or backing facts are unavailable.

The AQL path covers single- and multiple-producer queues, exact atomic
reservations, a deliberately unpublished hole, independent later publication,
MMIO doorbell ordering, an AQL barrier-AND packet ordered between dispatches,
scratch geometry, firmware control fields, memory borrowing, and private-segment
execution. A concurrent unbarriered control distinguishes barrier ordering from
mere serial scheduling. Each host producer waits for its own kernel completion
before reserving again, bounding this fixed-scratch workload to four concurrent
dispatches. Fresh single-producer queues also execute at low, normal, and high
priority. The recorded SDMA latency samples are observational; they establish no
latency or jitter guarantee, and the priority checks establish no relative
scheduling rate or fairness.

Isolated read-only SDMA faults passed under both lifetimes: each reached
sticky device loss and a fresh healthy workload completed afterward. Future
fault runs remain deliberately disruptive and should be isolated.
Dependency-bearing AQL barriers and barrier-OR need independent
qualification; dynamic scratch growth is beyond the fixed-backing GPU v1
contract. Multi-device activation and coherent SYSTEM-memory
access are implemented and unit tested, including common-VA selection, ordered
access records, distinct KFD device lists, repeated-consumer VM reuse,
partial retry progress,
device-to-device cache pairs, and dependency retention after ambiguous native
results. The current one-GPU host cannot qualify those paths in hardware. The
source implementation also admits LOCAL/VRAM peers only through cached direct
XGMI or kernel-validated PCIe routes, selects the physical owner independently
of caller order, and maps the backing through each distinct VM. Real multi-GPU
execution remains the topology gate, followed by broader cache-pair recipes,
kernel queues on other GPU targets, peer-device queue production,
live reset/unplug and cross-runtime stress, AArch64 GPU/cache execution, and
additional GPU targets. Cached loss publication, inherited-process rejection,
runtime non-interference, ambiguous cleanup, and the AArch64 source build are
covered without claiming those hardware gates.

The [alignment audit](docs/alignment-audit.md) records contract checks, planning
disposition, and verification performed after adopting AMDF.
