# rocJITsu DBI SuperCollider

This directory tracks the rocJITsu DBI MVP for SuperCollider-style LDS race
instrumentation on AMD RDNA4 / `gfx1201`.

The current implementation is a trap-first, HSA-tools-loaded proof path. It can
modify native `gfx1201` code objects at load time, patch compact native-DS IREE
kernels through inline padding, local NOP caves, or appended `.text` caves, and
patch selected likely group/LDS flat helper-function accesses in hip-moi. It is
not yet full SuperCollider race detection.

## Start Here

- [DESIGN.md](DESIGN.md): technical design and comparison with the
  SuperCollider paper, including flat/generic access handling.
- [USAGE.md](USAGE.md): commands, environment variables, and test runbook.
- [JAKUB-DEMO.md](JAKUB-DEMO.md): compact evidence packet from the one-hour
  Jakub demo pass.

## Current Demo Claim

With the default native-DS mode:

```sh
export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_DBI_SUPERCOLLIDER=1
export RJ_DBI_LOG=1
export RJ_DBI_SC_DELAY=2
export RJ_DBI_SC_REQUIRE_PATCH=1
```

the focused IREE WMMA ROCm/HIP e2e test has logged:

```text
kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
```

and passed. The broader IREE demo now also passes a 10-test patch-required
matrix covering narrow matmul, f16/f8/i8 TileAndFuse variants, DT f8, and
StableHLO stream-dot variants with `RJ_DBI_SC_DELAY_MODE=sleep`,
`RJ_DBI_SC_MAX_PATCHES=4`, and `RJ_DBI_SC_REQUIRE_PATCH=1`.

The same focused WMMA control passes with `RJ_DBI_SC_DELAY_MODE=sleep` and with
`RJ_DBI_SC_DELAY_MODE=sleep_var`.

## What Is Instrumented Today

The current check/trap proof paths cover:

- padded native LDS `ds_load_b{32,64,128}`,
- padded native LDS `ds_load_u16_d16` and `ds_load_u16_d16_hi`,
- padded native LDS `ds_store_b{32,64,128}`,
- compact native LDS sites through a local NOP cave when one is reachable, or an
  appended `.text` cave when that is the safe available placement,
- likely group/LDS `flat_load_b{32,64,128}`,
- likely group/LDS `flat_store_b{32,64,128}`.

Native DS sites can use enough trailing `s_nop 0` padding for an in-place
sequence, or reachable local NOP caves for compact sites, bounded by
`RJ_DBI_SC_MAX_PATCHES`. Flat/VFLAT sites can use trailing padding or, for one
selected site, a reachable local NOP cave. Ordinary hip-moi matmul helper code
has shown likely group/LDS flat sites rather than native `ds_*`, which is why
the flat path matters.

See [DESIGN.md](DESIGN.md) for the exact instruction policy and the current
address-space provenance heuristic.

## Main Gaps

- `RJ_DBI_SC_DELAY_MODE=sleep` emits `s_sleep N`, and
  `RJ_DBI_SC_DELAY_MODE=sleep_var` emits `s_sleep_var` from a scalar source
  operand. The remaining delay gap is randomized sampling policy, not the basic
  sleep instruction mechanism.
- `RJ_DBI_SC_MAX_PATCHES=N` can patch multiple native-DS check/trap sites in
  one code object, bounded by non-overlapping in-place ranges and distinct
  reachable local NOP caves.
- Trap is still the report mechanism. A report-buffer ABI is deferred.
- Current flat provenance is conservative and heuristic. `MaybeGroup` is useful
  for MVP bring-up, but it is not the same as a formal proof that an arbitrary
  flat access targets LDS.
- Current live flat check/trap mode patches one selected site per code object.
  Bounded multi-site instrumentation still needs a max-patches or sampling knob.
- Native DS d16 support currently covers `ds_load_u16_d16` and
  `ds_load_u16_d16_hi`; other 8/16-bit LDS forms remain deferred.

## Test Discipline

hip-moi is the inner loop. IREE is broader compatibility coverage and should be
run roughly once per work session. Keep GPU test parallelism at or below 8.
