# rocJITsu ConSan

ConSan is rocJITsu's DBI sanitizer work for AMD LDS/shared-memory race
instrumentation. It intercepts GPU code-object loads through the HSA tools
interface, inspects final native RDNA4 / `gfx1201` or CDNA4 / `gfx950` machine
code, and loads patched replacement code objects when instrumentation is
possible. ConSan patches the code object for the GPU that will execute it; it
does not translate between GPU architectures.

The current public flavors are:

```sh
RJ_CONSAN_FLAVOR=supercollider
RJ_CONSAN_FLAVOR=moi
```

Use `supercollider` first when sharing the current snapshot with a teammate. It
is the shortest useful path: redundant-access LDS/likely-group-flat checks,
configurable delay, and trap or marker-buffer reporting.

Use `moi` for structured memory-order instrumentation. MOI has
`record_replay`, `inline_shadow`, and `sampled` engines. On gfx1201 and gfx950,
their standard paths allocate or preserve scratch registers, owner/epoch and
3D workgroup identity, and scalar special state automatically; explicit
register variables remain debug overrides. Per-engine report buffers are
allocated automatically when needed. Startup identifies the frozen
conservative profile as `standard-v1`; advanced dynamic-record and ordering
paths remain explicit extensions because they change patch composition and
buffer layout. Sampled immediate checking is also opt-in, but it is an
implemented GPU-side adjacent-slot check rather than a future placeholder.

## Documents

- [TUTORIAL.md](TUTORIAL.md): team-facing commands for SuperCollider and MOI.
- [DESIGN.md](DESIGN.md): current architecture, implemented behavior, and
  explicit capability boundaries.
- [SPILLING.md](SPILLING.md): the R1 register allocator, gfx1201 and gfx950
  spill backends, ownership rules, provenance, and validation boundary.
- [USAGE.md](USAGE.md): detailed environment-variable and test runbook.
- [PLAN.md](PLAN.md): dependency DAG, completion evidence, and deferred
  multi-architecture branch.

## Quick Start

Build the HSA hook:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j8
```

Run a HIP/HSA workload through SuperCollider:

```sh
export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_CONSAN_FLAVOR=supercollider
export RJ_CONSAN_LOG=1
export RJ_CONSAN_DELAY_MODE=sleep
export RJ_CONSAN_DELAY=1
export RJ_CONSAN_MAX_PATCHES=1

./your-hip-or-hsa-program
```

Look for:

```text
ConSan summary ... patches=1 modified=true
```

Once a focused workload is known to contain supported LDS sites, add:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
```

That guard prevents a vacuous demo where the HSA hook loaded but no supported
site was patched.

## Current Capability Summary

SuperCollider currently supports:

- selected native LDS `ds_load_*` and `ds_store_*` instructions;
- selected likely group/LDS `flat_load_*` and `flat_store_*` instructions;
- inline padding, local NOP caves, and appended `.text` caves for patch
  placement;
- deterministic NOP, `s_sleep`, and `s_sleep_var` delay encodings;
- default `s_trap` reporting and optional one-word marker-buffer reporting.

MOI currently supports:

- `record_replay`: DBI access/barrier/atomic records plus host-side exact
  replay diagnostics;
- `inline_shadow`: direct GPU-side exact-shadow updates for supported native
  multi-cell and admitted group-flat LDS forms, compact diagnostics, barrier
  epoch controls, and narrow atomic ordering controls;
- `sampled`: runtime-qualified sampled watchpoint publication, host-side
  conflict scanning, and opt-in immediate checking.

The R1 resource path covers the current gfx1201 and gfx950 MOI probes: access,
barrier, and atomic sites share an owner-aware dead/fresh/spill planner, and
ordinary runs do not select register numbers. On gfx950's automatic
scalar-identity path, when a patch budget cannot cover every access site,
record/replay, sampled, and inline-shadow use the same stable resource
preference so a spill-free site is not hidden behind an earlier spill
candidate. The gfx1201 `standard-v1` profiles have passed the common 209-test
broad IREE tier. On gfx950, focused and guarded selected tiers are complete;
the 259-test SuperCollider inventory has no corruption, loader failure, or
timeout, with 257 ordinary passes and two typed `s_trap 0` sanitizer outcomes.
The three broad MOI profiles are tracked separately and must not be inferred
from that SuperCollider result.

## Acceptance And Next Work

The current gfx1201 snapshot passed 183 focused unit tests, 37 live rocJITsu
controls, 189 independent hip-moi semantic controls, 8 selected IREE tests per
profile, and 209 broad IREE tests per profile, with no timeout. A useful run
should show `patches=N modified=true`; focused MOI runs additionally use the
record or diagnostic guards documented in [TUTORIAL.md](TUTORIAL.md).

These results establish guarded DBI execution and broad output compatibility,
not universal opcode coverage or proof that a clean workload is race-free.
Native gfx950 implementation, spill/resource management, workgroup identity,
and selected semantic qualification are in place. Remaining gfx950 work is the
three broad MOI compatibility profiles and documented precision/coverage
extensions. Future native-target work still includes `gfx942` and `gfx1250`.
Narrow extensions such as more inline atomic forms remain capability
improvements rather than hidden acceptance requirements.
