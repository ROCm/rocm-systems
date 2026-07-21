# ConSan malformed-input containment contract

This document defines ConSan's malformed-input and containment contract. It is
deliberately separate from detection quality: a timeout, crash, reset, or
application failure is never a ConSan race diagnostic.

The contract applies to SuperCollider and to all three MOI engines:
`record_replay`, `inline_shadow`, and `sampled`. It covers the public transform
boundary, the HSA hook's install decision, and carefully contained execution of
structurally valid but semantically ill-formed GPU programs. Arbitrary
structurally malformed bytes must never be submitted to a GPU.

## Required outcomes

Every completed transform invocation produces exactly one typed,
transactional outcome:

| Transform outcome | Replacement bytes | Fail-open install | Fail-closed install |
| --- | --- | --- | --- |
| `ModifiedValid` | Nonempty, independently validated, and accompanied by a nonempty patch ledger | replacement | replacement |
| `Unchanged` | Empty | original | original |
| `Unsupported` | Empty | original | reject |
| `Invalid` | Empty | original | reject |

A non-`ModifiedValid` result with replacement bytes or patches is a contract
violation. A `ModifiedValid` result that fails final structural validation is a
contract violation and must be rejected under both policies. The loader must
make its decision from the typed result and final-validation bit; it must not
infer success from a nonempty byte vector.

`Unchanged` is not proof that an input is well formed. For example, a malformed
symbol record which is irrelevant to the selected transform may remain outside
the transformer's semantic validation boundary. Such an input retains an
explicit original-load outcome, not relabeled as rejection. Expanding the
semantic validation boundary is separate work; the contract must accurately
report what is and is not rejected today.

## Input and report-state boundaries

ConSan checks the ELF and code ranges it consumes before decoding or rewriting
them. Truncated headers, out-of-range sections or symbols, invalid kernel
descriptors, overflowing ranges, and impossible text-growth requests must
produce a typed unchanged, unsupported, or invalid outcome rather than a
partially transformed image. Structurally malformed bytes must never be
submitted to the GPU as a ConSan replacement.

The semantic boundary is narrower than a general-purpose ELF verifier.
`Unchanged` means only that the selected transform did not replace the image;
it is not a certificate that every unrelated part of the input is well formed.

Malformed or unstable report state is handled separately from malformed code
objects. Invalid versions, changed generations, impossible capacities,
malformed owner/epoch identities, publication collisions, and dropped records
make the analysis incomplete. They must never be interpreted as proof of a
clean execution.

## Hook and loader behavior

The HSA hook applies the transaction outcome consistently:

- fail-open loads the original for `Unchanged`, `Unsupported`, or `Invalid`;
- fail-closed rejects `Unsupported` and `Invalid`;
- only `ModifiedValid` may install replacement bytes; and
- a replacement that fails final structural validation is always rejected.

This decision is based on the typed result and final-validation bit, not merely
on whether a byte vector happens to be nonempty.

## Ill-formed GPU execution

Some programs are structurally valid code objects but contain semantically
ill-formed synchronization, such as divergent barriers or unmatched
split-barrier operations. ConSan cannot generally make execution of such a
program safe. A host-process timeout does not establish that an already
submitted kernel stopped, and a device that appears healthy afterward does not
turn the timeout into successful detection.

Run intentionally destructive experiments in an isolated process, with an
external timeout and device-health checks. Do not execute arbitrary malformed
bytes on a GPU.

## Optional unmatched-wait guard

Every flavor supports the narrow opt-in guard
`RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT=1`. It replaces only a statically unique
immediate wait that belongs to no bounded multi-event barrier sequence with
`s_endpgm`, and records an `inline-malformed-barrier-abort` patch. Bounded
non-adjacent pairs are associated before this decision.

Dynamic or ambiguous waits remain untouched rather than being guessed
malformed. Barrier fault injection disables the guard while mutation and
instrumentation are composed. This option is a containment aid for one
specific static shape, not a general hang detector or a race diagnostic.

## Current limitations

- There is no internal watchdog around arbitrary code-object transformation.
- Transactional host rejection does not prove containment of a kernel that was
  already submitted.
- The unmatched-wait guard covers only its stated unique immediate form.
- A timeout, signal, process failure, or GPU reset is not a ConSan diagnostic.
