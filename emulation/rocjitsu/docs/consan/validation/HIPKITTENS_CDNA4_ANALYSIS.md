# HipKittens BF16 startup fault on CDNA4

The remaining Default miss is an asynchronous-completion blind spot, not a
watchpoint-bank shortage. The original startup fault remains **unqualified**:
`max` with 128, 256, and 512 banks each passes the clean control and detects
0/8 faults with the corrected lane mapping. A reduced reproducer establishes
that Default can hide a missing producer VM wait and report zero conflicts.
The original GEMM mutation itself has not produced a numerical failure in the
native runs below; the reduced case proves the mechanism, not an observed
failure of that GEMM launch.

## Original mutation and barrier rounds

The retained code object is `fnv1a64:bff9c3ab1b70aea3`, kernel
`_Z8micro_tk13micro_globalsiii`, workload `-m 256 -n 256 -k 256`. The original
fault replaces only `s_barrier` at `.text+0x6d8` (static occurrence 1).

All eight waves issue eight initial direct-to-LDS loads. The prologue then has:

| Label | `.text` offset | Participants | Preceding completion condition |
| --- | --- | --- | --- |
| A | `0x6c0` | Waves 4–7 | No VM wait after the initial loads |
| B | `0x6d8` | All waves | `s_waitcnt vmcnt(4)` |
| C | `0x784` | All waves | Six more loads, then `s_waitcnt vmcnt(6)` |
| D | `0xad4` | All waves | First LDS reads, then `lgkmcnt(8)` |
| E | `0xb64` | All waves | `lgkmcnt(0)` and first MFMA group |

Static PCs do not identify dynamic workgroup barrier rounds. In the clean
kernel, low/high halves rendezvous at B/A, C/B, D/C, E/D, respectively.
Deleting B changes the first two rendezvous to C/A and D/C. All initial loads
have been **issued** before C/A, so an issue-based LDS history places them
before every first consumer read. However, the high half reaches A before its
VM completion wait. A low-half consumer may read a high-half producer's LDS
range before that copy completes. For example, low wave 1's first B-tile read
starts at the B-tile base plus 4096, in the range loaded by high wave 4.

The [CDNA4 ISA reference, S_BARRIER](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
explicitly requires a suitable wait before a barrier that protects outstanding
memory operations; the barrier itself does not drain those counters.

## Physical evidence

All runs use gfx950, venv TheRock, and the shared physical-GPU lock. Artifacts
are under `/home/benjacob/consan-default-repairs-20260924/`.

| Experiment | Result |
| --- | --- |
| Original clean GEMM, native, eight launches | Numerical pass 8/8 |
| Exact startup mutation, native, eight launches | Numerical pass 8/8 |
| Exact startup mutation, Default, 128/256/512 banks | Matching clean pass; 0/8 detections at each size |
| Separate retirement mutation at E, Default, 128 banks | Matching clean pass; 8/8 detections |
| Reduced copy, VM wait before barrier, native, three launches | 0/16,384 wrong values each |
| Reduced copy, VM wait after barrier, native, three launches | 16,384/16,384 wrong values each |
| Both reduced members under Default, 512 banks | 0 wrong values; 0 conflicts; complete access/barrier coverage |

`hipkittens-native-diagnostic/manifest.json` pins isolated executable copies.
The startup copy changes exactly the original four-byte barrier instruction to
`s_nop 0`; no other instruction, input, or oracle changes. Logs and results are
in that directory. The native retirement mutation also passes its numerical
oracle, illustrating why numerical success alone does not disprove a race.

The retirement control identifies an actual LDS read/write conflict: owner 4
reads at `.text+0xa34` while owner 0 writes at `.text+0xbac`, both covering
`[65536,65552)` in the same workgroup and epoch. That control is retained in
`hipkittens-bf16fp32-16x32/max-retirement-control-v3-banks-128/`. It does not
replace or qualify the original startup fault.

The reduced source is
[`cdna_async_completion.hip`](../../../tests/dbi/consan/reproducers/cdna_async_completion.hip).
Its two waves first initialize LDS and synchronize. The producer then issues a
direct-to-LDS load. The correct member waits before the publication barrier;
the broken member waits after it, while the consumer reads between the two
barriers. Both members end with a rendezvous and retain the same exact-value
oracle. `async-repro-native.log`, `async-repro-consan-0.log`, and
`async-repro-consan-1.log` retain the outputs. This is a manual diagnostic with
an intentionally failing native member, not a passing regression claim.

To reproduce, compile with the active TheRock `hipcc` (and the host C++ include
and library flags required by that installation), then run the executable
under `/tmp/rocjitsu-consan-destructive-gpu.lock`. Argument `1` selects the
correct member, `0` the broken member; the second argument is the workgroup
count (256 in these experiments). Instrumented runs use `RJ_CONSAN_PRESET=max`,
`RJ_CONSAN_WATCHPOINT_BANKS=512`, `RJ_CONSAN_POLICY=strict`, and `RJ_CONSAN_LOG=1`.

## Why Default misses it

`consan_access_emission.cpp` executes the direct-to-LDS guest instruction, then
emits the access probe. The probe claims its metadata slot with a returning
FLAT atomic and calls `append_global_atomic_wait`. On CDNA4,
`append_global_atomic_completion` emits `s_waitcnt vmcnt(0)`. That wait drains
both the metadata operation and the guest's outstanding copy. The instrumented
producer therefore reaches its barrier only after the copy has completed.

Separately, the causal record represents the write at the instruction's issue
epoch. It does not retain the guest copy's pending interval across barriers or
track which guest VM wait completes it. The access-order model consequently
finds no unordered pair. The reduced broken member's complete coverage verdict
means that all supported access/barrier sites were instrumented; it does not
prove coverage of asynchronous completion semantics. Waitcheck's preflight
also passes: this hazard is between waves, not an ordinary local register-use
dependency.

Simply removing the probe's wait is unsafe: the returned metadata value is
needed before the probe continues. Adding banks, sleeps, or a different fault
selector cannot repair the missing causal semantics. A full fix needs guest
VM-operation tracking independent of instrumentation waits, pending LDS-write
lifetimes across barriers, and completion proofs for nonzero VM counts and
control-flow joins. The original startup mutation must remain the acceptance
case, alongside the reduced correct/broken pair and the retirement control.
