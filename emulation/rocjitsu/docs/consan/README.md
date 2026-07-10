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

Use `moi` for structured memory-order instrumentation experiments. MOI already
has useful `record_replay`, `inline_shadow`, and `sampled` engines, but some
paths still require explicit prototype knobs for scratch registers, owner/epoch
state, and report buffers.

## Documents

- [TUTORIAL.md](TUTORIAL.md): team-facing commands for SuperCollider and MOI.
- [DESIGN.md](DESIGN.md): current architecture, implemented behavior, and
  explicit prototype gaps.
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
- `inline_shadow`: direct GPU-side exact-shadow updates for a narrow native LDS
  subset, compact diagnostics, barrier epoch controls, and narrow atomic
  ordering controls;
- `sampled`: direct sampled watchpoint publication plus host-side sampled
  conflict scanning.

The main prototype gap is resource management. Several MOI probes still depend
on explicit user-selected registers. The next major engineering phase is to
centralize scratch allocation and reuse or extend existing DBT spill/cave
utilities.
