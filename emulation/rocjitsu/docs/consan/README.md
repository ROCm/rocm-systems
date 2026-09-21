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

By default, ConSan samples memory accesses and collects bounded causal evidence
to diagnose races. Its alternative **SuperCollider** mode checks redundant
observations for value instability.

- `RJ_CONSAN_MODE=default`: sampled causal windows and conflict diagnostics;
- `RJ_CONSAN_MODE=supercollider`: redundant-access/read-back checking with an
  automatic non-trapping mismatch marker.

Both modes select every relevant static site they support and manage registers
and reporting automatically. ConSan barriers and atomics are on by default.
ConSan chooses its runtime sampling parameters automatically. Users do not
choose a patch count, register, report size, synchronization switch, or sampling
residue for ordinary runs.

## Quick start

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env HSA_TOOLS_LIB="$CONSAN_HOOK" ./application
```

Loading the hook is itself the activation action; no separate enable variable
is required. It selects ConSan by default. Add `RJ_CONSAN_LOG=1` for
instrumentation and completeness summaries.

For code objects not excluded by the kernel allowlist, the same hook runs
waitcheck over each supported original code object before ConSan DBI. It
reports missing waits or analysis failures, then continues into ConSan so
suspect kernels are still instrumented. No separate waitcheck HSA tool or
waitcheck environment settings are needed for a ConSan run.

For a focused program known to contain supported sites,
`RJ_CONSAN_POLICY=strict` defaults fail-closed and require-patch checks on; for
ConSan it also defaults automatic-record and forbid-overflow checks on. It does
not make race diagnostics fatal or replace inspection of the static coverage
summary.

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

For small repros, use `RJ_CONSAN_PRESET=higher`; use `max` to remove
workgroup and cell sampling. `low` trades coverage for lower recording overhead,
and `default` preserves standard behavior. See [ConSan presets](USAGE.md#presets)
for exact settings, overrides, and bounded-retention limitations.

## Documents

- [MODES.md](MODES.md): conceptual, phase-by-phase comparison of what the
  two modes do on the device, defer for later, and do on the host.
- [TUTORIAL.md](TUTORIAL.md): getting started on your own program.
- [USAGE.md](USAGE.md): everyday commands, presets, allowlists, and reading results.
- [EXPERT_CONTROLS.md](EXPERT_CONTROLS.md): detailed controls, resource limits,
  and validation interfaces.
- [DESIGN.md](DESIGN.md): architecture, implemented behavior, and semantic
  boundaries.
- [CAPABILITIES.md](CAPABILITIES.md): normative target-by-mode access,
  barrier, atomic, fence, and typed-exclusion matrix.
- Target qualification ledgers: [CDNA3 / gfx942](validation/STATUS_CDNA3.md),
  [CDNA4 / gfx950](validation/STATUS_CDNA4.md),
  [RDNA3 / gfx1100](validation/STATUS_RDNA3.md),
  [RDNA4 / gfx1201](validation/STATUS_RDNA4.md), and
  [CDNA5 / gfx1250](validation/STATUS_GFX1250.md).
- [VALIDATION.md](validation/VALIDATION.md): reproducible physical, simulator, and offline
  gates behind those ledgers.
- [BENCHMARK.md](benchmark/BENCHMARK.md): the separate reproducible performance
  contract, Aorta workload survey, and target-specific benchmark ledgers.
- [EMPIRICAL_METHODOLOGY.md](EMPIRICAL_METHODOLOGY.md): cross-cutting corpus,
  provenance, fault-detection, and recommendation principles.
- [SPILLING.md](SPILLING.md): ConSan register selection, ownership, private
  layout, and runtime integration.
- [AMDGPU register spilling](../spilling.md): reusable RocJitsu allocation and
  target-specific save/restore backends.
- [MALFORMED_INPUT.md](MALFORMED_INPUT.md): finite malformed-input and GPU
  containment contract.
