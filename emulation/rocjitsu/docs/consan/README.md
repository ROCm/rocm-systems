# rocJITsu ConSan

ConSan instruments AMD LDS/shared-memory behavior by intercepting HSA
code-object loads, inspecting final native machine code, and loading a patched
replacement when instrumentation is possible. ConSan has native support for
`gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`; it does not translate
between GPU ISAs.

The runtime agent supplies the active workgroup-LDS capacity, so target support
does not assume a fixed architectural size. Simulator and offline validation
use the selected RocJITsu JSON configuration as the source of truth for that
capacity.

ConSan exposes the SuperCollider flavor and three MOI engines. MOI stands for
**Memory-Ordering Instrumentation**.

- `RJ_CONSAN_MODE=supercollider`: redundant-access/read-back checking with an
  automatic non-trapping mismatch marker;
- `RJ_CONSAN_MODE=record-replay`: bounded
  records plus host replay;
- `RJ_CONSAN_MODE=sampled`: bounded statistical
  causal windows; and
- `RJ_CONSAN_MODE=inline-shadow`: supported-form
  exact GPU shadowing and attributed diagnostics.

The flavor and all three engines select every relevant site they support and
manage registers and reporting automatically. MOI barriers and atomics are on
by default. Sampled chooses its runtime sampling parameters automatically.
Users do not choose a patch count, register, report size, synchronization
switch, or sampling residue for ordinary runs.

## Quick start

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=sampled \
  ./application
```

Loading the hook activates MOI Record/Replay by default. On gfx1201, the
empirical recommendation is to select Sampled for ordinary barrier/LDS triage
and reserve Record/Replay for expert synchronization investigations. The coded
default has not yet changed. Add `RJ_CONSAN_LOG=1` for instrumentation and
completeness summaries.

When enabled, the same hook always runs waitcheck over each supported original
code object before ConSan DBI. It reports missing waits or analysis failures,
then continues into ConSan so suspect kernels are still instrumented. No
separate waitcheck HSA tool or waitcheck environment settings are needed for a
ConSan run.

For a focused program known to contain supported sites,
`RJ_CONSAN_POLICY=strict` prevents ineffective or incomplete instrumentation
from looking clean. It does not make race diagnostics fatal.

Look for transformed-byte, coverage, and completeness records:

```text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan summary ... patches=N modified=true
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

If the program's own correctness checks pass, ConSan preserved its result for
that run; this does not prove the program race-free. A failure, timeout, signal,
or GPU reset is not by itself a ConSan diagnostic.

## Documents

- [FLAVORS.md](FLAVORS.md): conceptual, phase-by-phase comparison of what the
  flavor and three engines do on the device, defer for later, and do on the
  host.
- [TUTORIAL.md](TUTORIAL.md): getting started on your own program.
- [USAGE.md](USAGE.md): public controls, defaults, coverage, and diagnostics.
- [DESIGN.md](DESIGN.md): architecture, implemented behavior, and semantic
  boundaries.
- [CAPABILITIES.md](CAPABILITIES.md): normative target-by-engine access,
  barrier, atomic, fence, and typed-exclusion matrix.
- Target qualification ledgers: [gfx942](STATUS_CDNA3.md),
  [gfx950](STATUS_CDNA4.md), [gfx1100](STATUS_RDNA3.md),
  [gfx1201](STATUS_RDNA4.md), and [gfx1250](STATUS_GFX1250.md).
- [VALIDATION.md](VALIDATION.md): reproducible physical, simulator, and offline
  gates behind those ledgers.
- [GFX1201_EMPIRICAL_STUDY.md](GFX1201_EMPIRICAL_STUDY.md): the audience-facing
  physical-gfx1201 recommendation comparing engine overhead, detection yield,
  and implementation complexity.
- [EMPIRICAL_METHODOLOGY.md](EMPIRICAL_METHODOLOGY.md): the reusable admission,
  GPU-timing, fault-detection, provenance, and recommendation contract.
- [GFX1201_EMPIRICAL_RESULTS.md](GFX1201_EMPIRICAL_RESULTS.md): generated
  performance, detection, structural, and complexity tables for that study.
- [SPILLING.md](SPILLING.md): ConSan register selection, ownership, private
  layout, and runtime integration.
- [AMDGPU register spilling](../spilling.md): reusable RocJitsu allocation and
  target-specific save/restore backends.
- [MALFORMED_INPUT.md](MALFORMED_INPUT.md): finite malformed-input and GPU
  containment contract.
