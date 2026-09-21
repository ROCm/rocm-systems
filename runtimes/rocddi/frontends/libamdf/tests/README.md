# AMDF validation

Current ROCm Systems source validation: 2026-09-24. The pinned dynamic
GPU CTS and registered SDMA results below were rerun on this source under
PROCESS and INSTANCE. The scratch-backed AQL example was rebuilt against
the current library and passed 50 fresh-process dynamic runs in each
lifetime. The native queue smoke created all three advertised families
and both advertised same-device producer mappings in each lifetime. A
GPU-published queue example then passed five fresh 64-packet runs for each
of AQL and SDMA under PROCESS and INSTANCE, including ring reuse, completion,
consumption, and result checks. Isolated read-only SDMA faults produced
sticky device loss under both lifetimes; fresh GPU work passed after each.
Private and host-visible LOCAL SDMA and PM4 workloads passed in both
lifetimes; the public paths checked CPU and GPU access plus directional cache
pairs. PM4 used conservative GCR barriers and passed 16 checked copies with
four private or eight public ring wraps in each lifetime.
Scratch-backed AQL execution also passed with WC LOCAL source and target
under both lifetimes, including directional cache-pair checks.
Earlier shared and static AQL runs remain evidence for the 2026-09-17 source.
Headers and upstream CTS are pinned to
`hrx-system@4aa34130de44c45d68a48575cebfd0ff0610c461`.

The source keeps implementation-neutral native mechanisms in the private
`rocddi` rlib and AMDF in a separate preloadable `cdylib`. The current Linux
backend selects an instance-owned secondary KFD context for INSTANCE lifetime
on UAPI 1.19 or newer; PROCESS continues to use the primary context. KFD
USERPTR storage remains unavailable through KFD in a secondary context.
INSTANCE AQL queues use coherent GTT ring storage there; DRM GEM USERPTR
registration uses that context's acquired render VM. Run the GPU
memory smoke in fresh processes with `AMDF_REQUIRE_GPU=1` and the `process` or
`instance` argument.

The 2026-09-24 source passed 309 workspace Rust tests, Clippy with warnings
denied, formatting, Rustdoc, and C memory smoke in CPU and GPU modes under
both lifetimes. The pinned dynamic GPU CTS ran all 63 cases per lifetime:
57 passed, 6 skipped, and 0 failed under PROCESS and INSTANCE. Registration
and independent shared-backing teardown passed in both. Registered-host
SDMA completed 64 checked copies and eight ring wraps in both lifetimes.
The 2026-09-23 source passed four pinned CTS CPU/passive combinations
(dynamic and shared linkage, PROCESS and INSTANCE), each with 39 passing
tests. The vendored C KFD header confirmed the CREATE_PROCESS request and
record size.
The qualified SYSTEM DMA-BUF import path combines KFD placement with DRM GEM
creation and same-device handle facts. PROCESS and INSTANCE each passed a
READ-only imported subrange SDMA copy, a child-process attachment,
identity-mismatch failure preservation, lifetime separation, and re-export.
It requires the DRM GEM handle-list and creation-info ioctls; drivers without
either leave IMPORT unadvertised. Foreign API and distinct-GPU imports remain
unadvertised.

## Reproducible checks

Run from `runtimes/rocddi`:

```sh
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

The C sources in `tests/abi` support direct C11 and C++20 compile and link
checks. The pinned HRX checkout supplies the upstream CTS; running it requires
C++20 and GoogleTest. GPU cases additionally require native device access.

## Migration baseline and recorded execution qualification

| Gate | Result |
|---|---|
| Migration baseline workspace unit tests | 296 passed: 37 libamdf, 113 libhsa, and 146 rocddi tests |
| Clippy, formatting, Rustdoc | Passed with warnings denied on the pinned Rust 1.98 toolchain |
| Rust 1.85 MSRV | Complete workspace and all targets/features passed `cargo check` |
| Architecture boundary | Both peer frontends depend on the private rocddi `rlib`; frontend ABI isolation and KFD ownership checks passed |
| Independent preload | `libamdf.so` and `libhsa_runtime64.so` each loaded and passed representative pre-initialization calls in separate fresh processes |
| AArch64 source gate | The complete workspace and all targets/features passed `cargo check`; `rocddi` and the `amdf` static library also completed code generation for `aarch64-unknown-linux-gnu` with Rust 1.85 |
| Imported headers / generated ABI | 80 aggregate layouts, 103 aliases/enums, 529 fields, 230 constants verified; vendored header hashes match the pinned ABI-v3 mirror |
| KFD UAPI | Installed ROCm-header and target vendored-header checks passed, including DMA-BUF and IPC records and requests |
| C11 / C++20 | Shared/static C smoke and C++ header checks passed |
| Dynamic exports | Exactly `amdf_query_api` |
| Current native memory smoke | Rebuilt shared library passed PROCESS and INSTANCE GPU memory on one GFX1201 endpoint. Both lifetimes qualify host registration. |
| Current pinned dynamic GPU CTS | PROCESS and INSTANCE: 57 passed, 6 skipped, 0 failed across 63 cases each. Exact registration access and independent shared-backing teardown passed under both lifetimes. |
| Current registered-host execution | PROCESS and INSTANCE each completed 64 checked SDMA copies and eight ring wraps from caller-owned registered pages. |
| Current LOCAL execution | PROCESS and INSTANCE each completed 64 private and 64 public LOCAL SDMA round trips, then 16 private and 16 public PM4 LOCAL copies. Both public paths checked CPU reads/writes and prospective and concrete directional cache pairs; PM4 wrapped its ring four or eight times. |
| Current qualified DMA-BUF import | PROCESS and INSTANCE each passed READ-only imported subrange SDMA, child-process import, identity failure preservation, independent re-export, and source teardown before imported use. |
| Recorded earlier native GPU gate | An earlier source revision passed shared/static SYSTEM/LOCAL/registered/DMA-BUF memory and queue smoke, including two fresh queues in every advertised PM4/AQL/SDMA family. Each linkage completed the recorded SDMA and AQL workloads; a concurrent control and barrier-AND sequence verified AQL barrier-bit ordering. |
| Current AQL execution | The checked GFX1201 code-object v4 example passed 50 fresh SYSTEM runs in each lifetime; each recreated five queues, completed 516 dispatches plus one barrier packet, and checked 32 ring wraps. LOCAL mode passed one run in each lifetime with WC host mapping, directional cache pairs, and 516 completed dispatches. |
| Current device-produced execution | PROCESS and INSTANCE each passed five fresh runs of 64 GPU-published SDMA packets (eight target ring wraps) and 64 GPU-published AQL packets (four target ring wraps). Completion, consumption, results, and cleanup passed. |
| Current live fault observation | An isolated SDMA write to GPU read-only SYSTEM memory produced sticky AMDF DEVICE_LOST and reset epoch 2 under PROCESS and INSTANCE. A fresh healthy GPU-produced SDMA workload passed after each fault process. |
| Earlier AQL stress | 50 consecutive shared and 50 consecutive static priority-and-barrier runs passed on the earlier source; every run completed all four 128-dispatch workloads, the publication-hole check, and both ordering sequences. |
| Migration baseline upstream CTS, CPU mode | All six linkage/lifetime configurations passed against the pinned ABI-v3 corpus. Counts are below. Native GPU acquisition cases were skipped or filtered. |
| Recorded upstream enumeration example | Unmodified `hrx-system/libamdf/examples/enumerate.c` compiled and ran against the rebuilt shared library on the recorded GPU host |

| Linkage | INSTANCE | PROCESS |
|---|---|---|
| Runtime loading | 39 passed, 24 skipped, 0 failed | 39 passed, 2 skipped, 0 failed |
| Shared linkage | 39 passed, 24 skipped, 0 failed | 39 passed, 2 skipped, 0 failed |
| Static linkage | 39 passed, 24 skipped, 0 failed | 39 passed, 2 skipped, 0 failed |

Each baseline INSTANCE configuration contains 63 tests. The baseline CPU-mode
PROCESS filter ran 41 tests, excluding cases that require native GPU
acquisition. Baseline INSTANCE skips reflect the then-unsupported GPU
activation; the two PROCESS skips reflect absent XDNA endpoints. The
XDNA-only and GPU/XDNA interoperability CTS remain excluded because they
require a compiled XDNA provider.

The current pinned dynamic GPU CTS skips two XDNA endpoint cases and three
two-GPU cases. Both lifetimes skip the test requiring registration to be absent;
both execute the registration and shared-backing interop cases. The pinned GPU
CTS does not exercise DMA-BUF import; the checked-in SDMA example provides its
native execution gate. Static and shared GPU CTS were not rebuilt for this
source revision.

The recorded 2026-09-15 GPU CTS used the earlier 52-test corpus. Each linkage
reported 37 passed and 15 skipped for INSTANCE, and 46 passed and 6 skipped for
PROCESS. Those counts remain hardware evidence for that revision, not a current
CTS corpus result.

The C memory smoke checks same-resource and distinct-resource pair queries, the
difference between unknown and mismatched physical identity, and unchanged
outputs on rejected pairs. C and Rust checks cover required write-back
cacheability and maintenance for registered caller pages. GPU registration
tests cover
feature reporting under both lifetimes, exact READ/RW/RX/RWX permissions, subpage
offsets, complete page-cover lengths, independent aligned GPU addresses,
preserved caller host addresses, qualified write-back cache behavior, invalid
pointer rejection, output preservation, cleanup ordering, and caller ownership.
Rust coverage additionally checks the qualified GFX1201 SYSTEM-memory
PM4/AQL/SDMA
host-to-device acquire and device-to-host release, rejection when an AQL
direction lacks its required operation, producer WRITE and consumer READ
requirements, missing host cache capability, empty and invalid ranges, and
asymmetric flush/invalidate recipes. Multi-device adapter tests cover ordered
access/address records, scratch selection by access ordinal, duplicate and
mismatched request rejection, and device-to-device SYSTEM release/acquire.

Native tests cover a lazy wipe-on-fork process marker, its getpid fallback,
and raw-fork inherited-owner rejection. They also cover serialized one-shot
runtime enable, interrupted enable retry, failed disable retry, foreign-runtime
contention without an unowned
disable, ambiguous activation cleanup without replay, inherited-instance
rejection before callbacks and locks, queue admission before backing allocation,
disable before KFD close, retained-owner shutdown, existing-device recreation,
and late activation of a distinct VM. Multi-VM memory tests cover the ordered
distinct KFD device list, repeated-consumer VM reuse, common-address membership,
partial map/unmap retry prefixes, and retention of every VM dependency after
ambiguous native results. Queue rings and PM4/AQL EOP and context-save storage
request GPU execute permission. PM4 tests cover exact target and option gating,
fixed ring and control offsets, native COMPUTE creation, and monotonic expansion
after multiple unobserved ring wraps. Device-producer tests cover opt-in
capability selection, host-only-to-device doorbell upgrade, same- and peer-VM
ring/index/doorbell addresses, common-VA and same-instance requirements, exact
doorbell BO flags, map/unmap ownership, clean unsupported rollback, retryable
teardown, public-device identity, and output preservation.
Scratch tests cover geometry and
encoding rejection, firmware-compatible queue-control fields, range/access/
reset validation, native-creation rollback, queue-lifetime borrowing, and
successful release after queue destruction. Native queue tests verify
secondary-VM AQL coherent GTT ring selection when KFD rejects USERPTR.
They also verify that the AQL control block encodes SINGLE as HSA queue type
1 and MULTI as type 0, and that low, normal, and high requests reach KFD
as priorities 0, 7, and 15. SDMA
continues to reject non-normal priorities. Before the EOP/context correction, the
independently validated kernel timed out through the AMDF queue. Matching
ROCr/libhsakmt's executable backing made that same path complete. An earlier
cold-process experiment also showed that deferring runtime enable until queue
creation stalled the first SDMA command; activation therefore remains part of
first device creation on this tuple.

Native memory tests verify the USERPTR allocation flag, page-aligned caller
address supplied to KFD, independent GPU-VA reservation, logical address offset,
exact permissions, and cleanup that never unmaps or frees caller pages.
Topology and multi-VM tests additionally cover directional direct-XGMI and
kernel-validated PCIe LOCAL routes, disabled and unknown link flags, malformed
source/count metadata, owner selection independent of caller order, and mapping
one LOCAL backing through every distinct qualified VM at one common address.
DMA-BUF tests additionally cover metadata-size retry, duplicated input
descriptors, imported-handle rollback and ambiguous copyout retention, full-BO
CPU/GPU mapping with a logical source offset, origin/placement/range rejection,
canonical file identity, close-on-exec export ownership, and re-export identity.

The upstream PROCESS CTS executes direct PM4 through runtime-loaded, shared,
and static providers. Both PM4 cases pass: one copies between exact SYSTEM
attachments, and the other repeats the workload while concurrently creating,
using, destroying, and recreating peer device handles. The stream uses native
type-3 COPY_DATA, WRITE_DATA, EVENT_WRITE, ACQUIRE_MEM, and NOP packets. Queue
status observes the separate completion token and expands the ring-relative KFD
read pointer into AMDF's monotonic dword frontier.

The upstream enumeration example reports one PCI `1002:7550` GPU with PM4, AQL,
and SDMA queue families. It remains passive. The separate
[execution examples](../examples/README.md) activated the GPU and used only
imported AMDF tables. SYSTEM SDMA verified publication, fence completion, queue
consumption, copied bytes, bounds, and ring reuse. The LOCAL workload verified
private device-local placement, exact read/write access, a usable GPU address,
rejected host mapping with output preservation, and 64 checked
SYSTEM-to-LOCAL-to-SYSTEM copies with untouched tails and eight ring wraps.
The registration workload verified subpage-offset source, target, and completion
buffers, exact access and metadata, independent GPU addresses, directional
cache pairs, 64 checked copies, eight ring wraps, and caller-page survival after
AMDF destruction. The DMA-BUF workload exported the middle page of a three-page
shareable RWX SYSTEM allocation, imported it with exact metadata and ownership
transfer, destroyed the original allocation while the import remained usable,
copied the imported bytes through SDMA, observed completion and consumption,
re-exported two independent descriptors, and explicitly released both. Invalid
descriptor, source-alignment, and export-range cases preserved their values and
outputs. AQL verified executable code and kernarg access, system-scope packet fences,
real private-segment scratch instructions, kernel completion, queue consumption,
copy-add results, untouched output, and ring reuse. Its fixed 5,120-byte scratch
allocation covered four waves and remained busy while borrowed by each queue.
Fresh single-producer queues executed 128 dispatches each at low, normal, and
high priority and reported the requested priority. The four-thread normal-
priority phase also verified exact atomic reservations, an unpublished four-
packet frontier, out-of-order publication behind an INVALID slot, recovery
after publishing that hole, and ordered x86-64 MMIO doorbells. Each producer
waited for its own kernel completion before reserving again, keeping at most
four scratch-backed dispatches active. All modes used child-before-parent
cleanup through shared and static linkage. Priority execution establishes no
relative service, fairness, or latency guarantee.

A fresh normal-priority queue then ran two ordering sequences. The control
sequence proved that a following dispatch could complete while a predecessor
remained blocked on a host-released gate. The second sequence inserted a zero-
dependency barrier-AND packet with the barrier bit between those dispatches; the
follower remained incomplete for a 10 ms observation interval and ran only
after the predecessor was released. Both dispatch results and the five-packet
consumed frontier were checked. This does not qualify barrier-OR or nonzero
signal dependencies.

After bounding each producer by its prior kernel completion, 50 consecutive
shared and 50 consecutive static priority-and-barrier AQL example runs passed.
Each run recreated five queues, completed 516 dispatches plus one barrier packet,
and checked 32 aggregate ring wraps. This stress result is a correctness
observation on the recorded tuple, not a scheduling or performance guarantee.

Observed publication-to-completion-and-consumption samples were:

| Linkage | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Shared | 21,372 ns | 24,787 ns | 119,049 ns |
| Static | 21,042 ns | 22,644 ns | 118,247 ns |

Each run completed 64 copies with lengths from 1 through 65,535 bytes and eight
wraps of a 1 KiB ring. These are correctness-run observations, not latency or
jitter guarantees.

Host: Linux `6.17.0-23-generic`, x86-64, KFD UAPI 1.23, GPU target `120001`
(gfx1201), GPU ID `33844`, PCI device ID `0x7550`, render minor `128`, SDMA IP
7.0.1. The kernel source used as supporting lifecycle evidence is
`amdgpu@48f2f4486a8dc6122195fa8755b6a1a6ee2ad3e2`; exact compatibility with the
running distribution kernel is not established.

This qualifies the direct host-produced PM4 SYSTEM-memory copy and concurrent
device-recreation workload, PM4/SDMA kernel command execution on GFX1201, the
PROCESS shared-backing interop case, the host-produced SDMA SYSTEM-memory copy, the
private and host-visible LOCAL SDMA and PM4 round trips, registered-host
SDMA copies,
same-provider SYSTEM DMA-BUF subrange transport, low/normal/high single-producer
plus normal-priority multiple-producer
fixed-scratch AQL SYSTEM- and LOCAL-memory dispatch, zero-dependency barrier-AND
ordering, and same-device GPU-produced AQL and SDMA queue execution.
Dependency-bearing barriers, barrier-OR, dynamic scratch management, live
reset and unplug, cross-runtime stress, multi-device hardware execution,
AArch64 GPU execution and cache behavior, LOCAL/VRAM peer execution and cache transitions,
foreign or LOCAL external memory, arbitrary subpage DMA-BUF offsets, additional
GPU targets, and performance guarantees remain unqualified. Distinct-device VM
activation, coherent SYSTEM-memory access, and topology-qualified LOCAL peer
mapping are unit tested; this host exposes only one GPU. The read-only fault
probe passed under both native lifetimes, with a healthy workload after each.

## Historical native qualification, 2026-09-14

These results were recorded before the source directory and Cargo packages
were renamed, against the working-tree migration over
`a0d690c793ff7897d73290fb5a113f5b0ba0a1cc`. They do not represent
a fresh GPU run of the alignment corrections. Repeating these results requires
the pinned CTS checkout and native device access.

| Gate | Result |
|---|---|
| Rust 1.85 unit tests | 50 core and 18 adapter tests passed |
| Clippy, formatting, Rustdoc | Passed with warnings denied where applicable |
| Imported headers / generated ABI | 74 aggregate layouts, 98 aliases/enums, 502 fields, 220 constants verified by compiled C/Rust probes |
| KFD UAPI | Installed-header and local kernel-source-header checks passed, including exact permission flags |
| C11 / C++17 | Shared/static C execution and C++ header checks passed |
| Dynamic exports | Exactly `amdf_query_api` |
| Native GPU smoke | SYSTEM allocation/mapping/cache/cleanup and two fresh simultaneous queues in each advertised AQL/SDMA family passed, with shared and static linkage |
| Upstream CTS | All six linkage/lifetime combinations passed; counts below |

| Linkage | INSTANCE | PROCESS |
|---|---|---|
| Runtime loading | 37 passed, 15 skipped, 0 failed | 46 passed, 6 skipped, 0 failed |
| Shared linkage | 37 passed, 15 skipped, 0 failed | 46 passed, 6 skipped, 0 failed |
| Static linkage | 37 passed, 15 skipped, 0 failed | 46 passed, 6 skipped, 0 failed |

Each combination contains 52 tests. Skips reflect absent XDNA endpoints,
unimplemented GPU registration, unsupported INSTANCE GPU activation, and
GFX1151-specific tests on this GFX1201 host. The XDNA-only and GPU/XDNA interoperability CTS, whose configuration
requires a compiled XDNA provider, are excluded; adapter tests verify the
extension's explicit absence. Applicable GPU memory tests exercised SYSTEM
and LOCAL construction with READ, RW, RX, and RWX requirements.

Host: Linux `6.17.0-23-generic`, x86_64, GPU target `120001` (gfx1201),
PCI device ID `0x7550`, render minor `128`. The C queue smoke observes empty
queues; it does not submit packets or establish execution/visibility support.
No reset, unplug, AArch64, peer, or performance qualification is claimed.

## Historical source layout validation, 2026-09-14

On 2026-09-14, removed the previous build outputs and rebuilt from `libamdf/`
with packages `amdf` and `amdf-native`. Both crates are implementation packages
with Cargo publication disabled; consumers use the C ABI.

- Rust 1.85: all 68 tests passed (18 C adapter, 50 native implementation).
- Formatting, strict Clippy, and Rustdoc passed.
- Imported-header hashes and generated C/Rust layouts passed with the counts
  above. C11 shared/static smoke and C++17 syntax checks passed.
- The rebuilt shared library exports exactly `amdf_query_api`.
- The then-current CTS gate passed in dynamic, shared, and static modes.
  Each mode ran 52 INSTANCE tests (37 passed, 15 skipped) and 39 PROCESS tests
  (37 passed, 2 skipped), with no failures. This run used the script's default
  CPU mode; GPU acquisition tests were filtered from PROCESS runs.
- Source, documentation, and active plans contain no retired API name or
  broken relative Markdown file links.

The native GPU results in the preceding section were not rerun for the rename.
The latest CTS XML files have since been replaced by the 2026-09-15 CPU run.

## Contract corrections covered by the migration

- Short structure prefixes are checked before complete records are read;
  enlarged outputs preserve their headers and trailing bytes.
- Output-array validation never constructs initialized Rust references over
  uninitialized C output elements.
- Metadata allocation failures preserve outputs and release only acquired
  private ownership, using the caller's callbacks throughout.
- Queue mappings and differing producer/consumer frontiers block destruction
  before mutation; AQL reservations can precede release publication.
  Partial native teardown revokes cached transport access and preserves retry.
- Released native IDs and consumed file descriptors are never replayed.
  Ambiguous outcomes retain native dependencies. If those dependencies retain
  callback-backed metadata, instance destruction fails and preserves its owner;
  PROCESS lifetime does not extend the caller allocator lifetime implicitly.
- Exact device permissions reach KFD without widening; CPU permissions remain
  separately described by host mapping information.
- SYSTEM GTT host mappings are write-back. KFD UNCACHED is a GPU cache property,
  so it does not select an uncached host recipe. WC mappings use a real MFENCE
  operation, including in host-to-host pair descriptions.
- Host-visible LOCAL profiles require a qualified native cache mode. Device
  cache-pair queries fail without publishing incomplete transition recipes.

The GTT cache interpretation was traced through local kernel source
`amdgpu@48f2f4486a8dc6122195fa8755b6a1a6ee2ad3e2`: KFD GTT allocation does not
set CPU_GTT_USWC, and TTM selects cached CPU protection in its absence. That
source checkout is supporting evidence, not a proven revision match for the
running distribution kernel. Public VRAM's WC recipe is restricted to the
qualified ordinary GFX10.1–12.0 path; CPU-connected XGMI variants need separate
qualification.
