# rocJITsu ConSan

ConSan is rocJITsu's DBI sanitizer work for AMD LDS/shared-memory race
instrumentation. It intercepts GPU code-object loads through the HSA tools
interface, inspects final native RDNA4 / `gfx1201` machine code, and loads
patched replacement code objects when instrumentation is possible.

The current public flavors are:

```sh
RJ_CONSAN_FLAVOR=supercollider
RJ_CONSAN_FLAVOR=moi
```

Use `supercollider` first when sharing the current snapshot with a teammate. It
is the shortest useful path: redundant-access LDS/likely-group-flat checks,
configurable delay, and trap or marker-buffer reporting.

Use `moi` for structured memory-order instrumentation. MOI has
`record_replay`, `inline_shadow`, and `sampled` engines. On gfx1201 their
standard paths now allocate or preserve scratch registers, owner/epoch state,
and scalar special state automatically; explicit register variables remain
debug overrides. Per-engine report buffers are allocated automatically when
needed. Startup identifies the frozen conservative profile as `standard-v1`;
advanced dynamic-record, ordering, and immediate-sampled paths remain explicit
extensions because they change patch composition and buffer layout.

## Documents

- [TUTORIAL.md](TUTORIAL.md): team-facing commands for SuperCollider and MOI.
- [DESIGN.md](DESIGN.md): current architecture, implemented behavior, and
  explicit prototype gaps.
- [SPILLING.md](SPILLING.md): the R1 register allocator, gfx1201 spill backend,
  ownership rules, provenance, and validation boundary.
- [USAGE.md](USAGE.md): detailed environment-variable and test runbook.
- [PLAN.md](PLAN.md): near-future work DAG from the current baseline.

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

The R1 resource path is complete for the current gfx1201 MOI probes: access,
barrier, and atomic sites share an owner-aware dead/fresh/spill planner, and
ordinary runs do not select register numbers. The gfx1201 `standard-v1`
profiles have passed the common 209-test broad IREE tier; remaining work is
native-target breadth and explicitly documented precision/coverage extensions.
