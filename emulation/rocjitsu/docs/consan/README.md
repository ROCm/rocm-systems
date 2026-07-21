# rocJITsu ConSan

ConSan instruments AMD LDS/shared-memory behavior by intercepting HSA
code-object loads, inspecting final native machine code, and loading a patched
replacement when instrumentation is possible. ConSan has native support for
`gfx950`, `gfx1201`, and `gfx1250`; it does not translate between GPU ISAs.

ConSan exposes the SuperCollider flavor and three MOI engines. MOI stands for
**Memory-Ordering Instrumentation**.

- `RJ_CONSAN_FLAVOR=supercollider`: redundant-access/read-back checking with an
  automatic non-trapping mismatch marker;
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=record_replay`: bounded
  records plus host replay;
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=sampled`: bounded statistical
  causal windows; and
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=inline_shadow`: supported-form
  exact GPU shadowing and attributed diagnostics.

The flavor and all three engines select every relevant site they support and
manage registers and reporting automatically. MOI barriers and atomics are on
by default. Sampled chooses its runtime sampling parameters automatically.
Users do not choose a patch count, register, report size, synchronization
switch, or sampling residue for ordinary runs.

## Quick start

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_ENABLE=1 \
  RJ_CONSAN_LOG=1 \
  ./application
```

`RJ_CONSAN_ENABLE=1` selects MOI Record/Replay by default. It is the
recommended starting engine and provides an inspectable host-side model. Its
retained dynamic history is bounded, so a clean replay is not proof of race
freedom.

Loading the hook without `RJ_CONSAN_ENABLE=1` remains inert. Flavor and engine
variables select the analysis but do not enable ConSan by themselves.

When enabled, the same hook always runs waitcheck over each supported original
code object before ConSan DBI. It reports missing waits or analysis failures,
then continues into ConSan so suspect kernels are still instrumented. No
separate waitcheck HSA tool or waitcheck environment settings are needed for a
ConSan run.

For a focused program known to contain supported sites, the self-checks
`RJ_CONSAN_FAIL_CLOSED=1`, `RJ_CONSAN_REQUIRE_PATCH=1`, and
`RJ_CONSAN_MOI_REQUIRE_RECORDS=1` prevent ineffective instrumentation from
looking clean. These are assertions, not tuning controls.

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
- [SPILLING.md](SPILLING.md): ConSan register selection, ownership, private
  layout, and runtime integration.
- [AMDGPU register spilling](../spilling.md): reusable RocJitsu allocation and
  target-specific save/restore backends.
- [MALFORMED_INPUT.md](MALFORMED_INPUT.md): finite malformed-input and GPU
  containment contract.
