# ConSan RDNA3 (`gfx1100`) status

This is the target-native `gfx1100` qualification ledger. It records only
evidence produced by gfx1100 code objects on the physical Radeon Pro W7900 in
this workspace; it does not inherit coverage, behavior, or timing from an
RDNA4 target. The normative semantic boundary is the
[capability matrix](CAPABILITIES.md).

## Current qualification

The compact fixture uses native gfx1100 wave32 LDS instructions, a singleton
workgroup barrier, and target-specific code caves. Its clean case has two
waves communicate through LDS across the barrier. Its conflict case has both
waves write the same per-lane LDS cells without synchronization.

| Gate | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | --- | --- | --- | --- |
| Clean exact output | Passed; required a native patch and preserved every output word | Passed; required visible records, forbade diagnostics, and preserved every output word | Passed; required visible records, forbade diagnostics, and preserved every output word | Passed; required visible records, forbade diagnostics, and preserved every output word |
| Conflict sensitivity | The compact gate proves mutation and output containment; it does not claim causal race attribution | Not used as the compact conflict oracle | Not used as the compact conflict oracle | Passed; the same-site two-wave conflict produced an attributed diagnostic with distinct resident-wave owners and overlapping LDS ranges |
| Latest test time | 0.09 seconds clean; 0.10 seconds all-supported-site pass | 0.10 seconds | 0.10 seconds | 0.10 seconds clean; 0.10 seconds conflict |

The complete physical gate is `ConSanGfx1100Physical.*`. At the current
checkpoint all eight cases pass in 2.28 seconds, including cold baseline
startup:

- baseline exact output;
- SuperCollider clean mutation;
- a no-filter SuperCollider pass that patches every supported site in the
  fixture code object;
- Record/Replay, Sampled, and Inline Shadow clean execution;
- required Inline Shadow conflict attribution; and
- an uninstrumented post-run health check.

CTest resolves the physical gfx1100 agent UUID from `rocminfo` at configure
time and scopes each process with that stable identifier. The gate does not
assume a device ordinal. Runtime instrumentation takes LDS capacity from that
active agent; the gfx1100 JSON is used only for simulator and offline work.

The complementary `ConSanGfx1100Sim.*` gate runs the same native gfx1100 code
objects through RocJITsu with `gfx1100_w7900.json`. Its five compact cases
cover a no-filter all-supported-site SuperCollider pass, clean Record/Replay,
Sampled, and Inline Shadow execution, and required Inline Shadow conflict
attribution. This simulator evidence uses JSON-declared LDS capacity and is
reported separately from the physical results above.

## Semantic boundary

- Native LDS, admitted group-FLAT, and singleton `s_barrier` forms have
  target-specific RDNA3 decoding and lowering in all four engines.
- Inline Shadow uses the gfx11 `HW_ID1` resident-wave slice for wave-uniform
  ownership. The physical conflict case distinguishes two waves as owners 1
  and 513; this is architecture state, not a preserved guest SGPR range.
- The complete compiler-emitted gfx11 acquire sequence is recognized. A
  release is not inferred from an atomic instruction that lacks sufficient
  ordering evidence. Missing, reversed, or interrupted cache-operation pairs
  remain unassociated.
- gfx11 has no qualified dispatch-ID preload/user-SGPR contract. ConSan uses
  the established RDNA code-object/report identity literal instead of
  reserving a guest SGPR pair. This separates loaded-image/report generations;
  it is not a claim of a hardware-unique token for simultaneous launches of
  the same loaded image.
- Native 96-bit LDS, cluster barriers, and ordered LDS atomics are not claimed
  for the current gfx1100 subset.
- Register and spill paths use RDNA3 encodings for fixed private frames,
  runtime-selected dynamic frames, scalar composition, and borrowed-register
  bootstrap. Host tests also cross the 33-barrier Record/Replay dense-router
  threshold. Unrepresentable resource or instruction forms still fail closed.

## Remaining breadth

This checkpoint qualifies the target plumbing and compact physical behavior;
it is not a broad application campaign. Larger gfx1100 workloads, warm
overhead distributions, and a retained multi-application fault campaign remain
unassessed and must not borrow status from the gfx1201 ledger.

The bring-up also retains a no-regression gate: the complete host ConSan suite,
the generated capability-documentation check, physical gfx1201 smoke cases,
and compact gfx942, gfx950, gfx1100, and gfx1250 simulator slices must remain
green.
