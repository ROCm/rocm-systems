# AMDF implementation and planning alignment

Updated on 2026-09-24 for the ROCm Systems runtime workspace.

The audit covers the public headers, Rust provider, ABI tests, and tracked
design and contract documents. Its
acceptance criterion is an unchanged AMDF frontend over the private rocddi
core. Private Rust types and caller policy must not become competing public
APIs or imply capabilities that the provider has not qualified.

## Sources and authority

The seven [vendored headers](../../../../api-headers/README.md) are synchronized
from `hrx-system@4aa34130de44c45d68a48575cebfd0ff0610c461` with only the approved
AMD copyright and MIT license preamble substitution. These headers define the ABI; the
[support map](api-support.md) describes what this implementation supports. The
upstream CTS and consumer sources were inspected at that same revision.

## Requirements and evidence

| Requirement | Result and evidence |
|---|---|
| Preserve the AMDF public ABI | Imported AMDF headers remain libamdf's sole ABI source. Generated bindings and compiled layout probes match them; the shared library exports only `amdf_query_api`. libamdf and rocddi are private Rust packages. |
| Keep API frontends independently preloadable | libamdf and libhsa build separate `cdylib` artifacts, link the private rocddi `rlib`, and load independently through `LD_PRELOAD`. Same-process coexistence is not claimed while each artifact contains separate process-global native state. |
| Account for all callable services | The [support map](api-support.md) covers all 38 core ABI-v3 and six GPU-v1 table slots, including validation-only and unsupported paths. XDNA is absent. Table presence does not qualify every endpoint or request. |
| Preserve AMDF memory identity and cache semantics | [Memory implementation](../src/memory.rs) distinguishes mismatched physical identity from unavailable identity, requires the selected registration cacheability, and qualifies concrete and prospective write-back visibility recipes. C and Rust regression coverage exercises these distinctions and output preservation. |
| Establish complete ordered access sets | Multi-device SYSTEM CREATE and REGISTER use one common VA, pass the ordered distinct GPU-ID list through mapping retries, reuse a mapping for repeated consumers of one VM, and publish one immutable access record per requested device. LOCAL CREATE additionally requires the physical owner and a cached directional direct-XGMI or validated-PCIe route for every peer, allocates on the owner independently of request order, and maps the same backing through each distinct VM. SYSTEM IMPORT is qualified for one native VM; distinct-GPU IMPORT remains unadvertised. |
| Keep device production explicit | AQL and SDMA families advertise device production, but a queue acquires the KFD doorbell BO only when creation requires that capability. The owner mapping is immediate. A peer mapping requires the same instance, common VA coverage, and successful KFD attachment of the ring, indices, and doorbell. Hardware execution remains a separate gate. |
| Keep ownership and control lightweight | [Shared boundary rules](../src/support.rs) explain caller provenance, borrowed parents, destruction serialization, and output publication. The private rocddi [Driver](../../../src/driver.rs) owns native control; cached queries need no native call or global registry. |
| Honor method-level cost contracts | API-table negotiation and cached endpoint, family, scope, device, memory, mapping, and address queries use retained immutable or atomic state without allocation, locks, lazy initialization, or ownership-counter updates. Pair queries directly compose the two supplied sites. User status is the documented native-observation path; wait calls are the explicit synchronization path. |
| Publish exact queue encodings | Family format features propagate unchanged through created queue and mapping information. GFX1201 PM4 reports ACQUIRE_MEM GCR, SDMA 7.0.1 reports GCR plus explicit-system FENCE, and AQL reports the zero baseline. |
| Separate queue transport from application completion | [Queue documentation](../README.md) distinguishes producer reservations, release publication, consumption, and application resource lifetime. The GFX1201 multiple-producer workload observes unpublished reservations and later packets blocked behind an INVALID hole. |
| Keep caller services outside the DDI | Pool reuse, task graphs, scheduling policy, recording, and recovery remain in API frontends or their consumers. There is no native task or timeline API. |
| Distinguish tests from capability claims | The [validation record](../tests/README.md) separates CPU ABI execution, GPU construction checks, and hardware workload execution. |
| Define recovery and coexistence failures | Cached native loss advances reset epochs without adding work to metadata queries. Inherited instances reject work before callbacks or locks. Foreign runtime contention is not disabled by this provider, while ambiguous activation requires cleanup without replay. |
| Keep target claims evidence-based | Linux AArch64 passes a pinned-MSRV workspace source check and code generation for both Rust libraries. Native C linking, cache recipes, native transport, and GPU execution remain unqualified until target-specific evidence exists. |

## Contract corrections

`memory_query_pair_info` distinguishes valid mismatched physical identities
from unavailable identities. It rejects different instances and resources
undergoing teardown and preserves output bytes on failure. Shared backing still
needs qualified producer-release and consumer-acquire recipes before the
provider can publish a complete pair description. The GFX1201 PM4, AQL, and
SDMA families publish those SYSTEM-memory recipes only on their
execution-qualified native paths.

REGISTER profiles require callers to declare the source mapping's cache class.
The current CPU and Linux GPU paths accept qualified write-back pages. GPU
REGISTER maps the complete caller page cover through KFD or DRM, preserves the logical
subrange and host address, obtains an independent GPU address, and uses the
established write-back SYSTEM-memory recipes. Rust and C tests check these
metadata, ownership, permission, cleanup, and transition distinctions.

Supported one- and multi-device SYSTEM CREATE profiles expose same-provider GTT
DMA-BUF export. One-native-VM SYSTEM IMPORT combines KFD placement with DRM GEM
creation flags and same-device handle identity, then establishes exact GPU PTE
permissions and an independently owned write-back host view. It consumes the
move-owned value only after success. Unsupported source classes and failed
imports preserve the external value and output. Distinct-GPU and LOCAL imports
remain unadvertised.

Repeated entry-point safety comments were replaced with shared boundary
obligations and comments explaining actual ownership transitions. Rustfmt also
collapsed redundant closure blocks; those formatting changes add no behavior.

## Planning disposition

Local plans and historical evidence are non-authoritative inputs and are not
required to build or interpret the tracked API. They remain outside the
portable runtime migration set. The support map, this record, and validation
results retain the current decisions needed by a fresh checkout.

## Verification and next gates

The current workspace contains 308 passing Rust tests. The recorded AMDF
checks also include formatting, strict Clippy and Rustdoc, imported-header
and generated-layout verification, shared/static C smoke, C++20 header checks,
export checks, the AArch64 source gate, and historical CTS linkage/lifetime
configurations. The 2026-09-24 native memory smoke and pinned dynamic GPU CTS
passed under PROCESS and INSTANCE on the current source.
The 2026-09-17 AMDF runs contain 62 tests per configuration: each INSTANCE
run passed 38 and skipped 24, while each PROCESS
run passed 51 and skipped 11. The PROCESS runs include the direct-PM4 copy and
concurrent device-recreation workloads. Exact commands, counts, and hardware
limits are in the [validation record](../tests/README.md).

The GFX1201 workloads activate a device through the imported tables and check
64 SYSTEM-memory SDMA copies, 64 private and 64 host-visible LOCAL SDMA
round trips under both lifetimes, 64 registered-host SDMA copies, and 384 single-producer plus 128 four-thread
multiple-producer scratch-backed AQL copy-add dispatches per linkage. The AQL
workload also passed with WC LOCAL source and target in both lifetimes. The
registered workload verifies subpage-offset caller buffers, page-cover metadata,
independent GPU addresses, exact access, directional cache pairs, eight ring
wraps, and caller ownership after destruction. The LOCAL workload verifies private-map rejection and public WC mapping,
exact access, prospective and concrete cache pairs, CPU and SDMA access,
8 or 16 ring wraps, completion, consumption, results, untouched tails, and
cleanup.
AQL additionally covers low/normal/high queue creation and execution, exact KFD
priority values, fixed private-segment execution, exact single/multiple HSA
queue-control types, atomic reservation uniqueness, an unpublished frontier,
out-of-order later publication, MMIO doorbell ordering, geometry validation,
and queue-lifetime scratch borrowing. A separate two-dispatch control proves
concurrent progress without a barrier; inserting a zero-dependency barrier-AND
with the barrier bit blocks the follower until the predecessor completes.
Priority execution does not establish a relative service, fairness, or latency
guarantee.

Remaining gates include deliberate fault execution, dependency-bearing AQL
barriers, barrier-OR, additional device cache-pair recipes, foreign and LOCAL
external-memory interop, LOCAL/VRAM peer hardware execution, kernel queues,
live reset and unplug, cross-runtime stress, additional targets, and AArch64 GPU
execution. Same- and peer-device production, distinct-device activation,
SYSTEM-memory access, and source/unit-qualified LOCAL peer routing lack hardware
qualification on the current host. Each capability must remain unadvertised or
reject unsupported requests until its full execution and failure paths are
qualified.
