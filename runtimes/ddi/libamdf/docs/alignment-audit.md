<!-- SPDX-License-Identifier: MIT -->

# AMDF implementation audit

This audit records the relationship between the public headers, the AMDF
frontend, the rocddi core, and the checked qualification. The headers define
the ABI; private Rust types do not extend that contract.

## Sources and authority

The seven [vendored headers](../../../../api-headers/README.md) are synchronized
from `hrx-system@4aa34130de44c45d68a48575cebfd0ff0610c461` with only the approved
AMD copyright and MIT license preamble substitution. These headers define the ABI; the
[support map](api-support.md) describes what this implementation supports. The
upstream CTS and consumer sources were inspected at that same revision.

## Requirements and evidence

| Requirement | Result and evidence |
|---|---|
| Preserve the AMDF public ABI | Imported AMDF headers remain libamdf's sole ABI source. Generated bindings and compiled layout probes match them; `amdf_query_api` is the AMDF entry point in the shared image. libamdf and rocddi are private Rust packages. |
| Keep API frontends independently preloadable | The AMDF and HSA library names resolve to one combined shared image. Either name may be loaded alone or both may be loaded in one process; both ABIs then use one rocddi process context. The AMDF static archive is separate. |
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

## Verification and remaining limits

The [validation record](../tests/README.md) gives runnable source and ABI
checks and identifies the GFX1201 binary used for hardware results. The
[support map](api-support.md) states which API requests are implemented and
which remain unadvertised. Source and ABI checks do not qualify device
coherence, firmware behavior, or a second platform backend.

The current backend supports Linux x86-64 and AArch64 builds; GPU execution has
been exercised on one x86-64 GFX1201 host. Peer-device execution, live reset
and unplug, AArch64 GPU execution, and cross-runtime stress still need
hardware qualification. The combined shared image has one rocddi process
context for both frontend ABIs. The separately linked AMDF static archive
does not share that context.
