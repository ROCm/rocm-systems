# Known expected failures (XFAIL).
#
# These tests run but are expected to fail.  CTest treats them as:
#   - PASS if the test exits non-zero (still failing as expected)
#   - FAIL if the test exits zero   (unexpectedly fixed -- update this file!)
#
# When you fix one of these, remove the corresponding set_tests_properties
# line so the test is expected to pass again.

# 128x128-tile matmul family — PARTIAL GRADUATION under --enable-
# writelane-rewrite.
#
# Status after the post-graduation symmetry fix (see
# hotswap/docs/wave-size-translation.md §5.6.3 "writelane/readlane
# symmetry" and rewrite_cross_lane_divergent.{hpp,cpp}):
#
#   * Gfx1250Gpu.Matmul64x64                       — PASS (no
#     divergent writelane/readlane sites; rewrite no-op).
#   * Gfx1250Gpu.Matmul128x128_1tile_UniformDiag   — PASS (bit-
#     exact: A=B=1.0 random-independent pattern sensitises only the
#     address layout that the rewrite makes consistent).
#   * Gfx1250Gpu.Matmul128x128_1tile  (random)     — XFAIL (this
#     file).
#   * Gfx1250Gpu.Matmul128x128        (random 2x2) — XFAIL (this
#     file).
#
# Why the random-input tests still fail:
#
# The initial graduation (commits b13cdb1e96 + 2d8415f798) faulted
# the kernel at dispatch with HSA_STATUS_ERROR_MEMORY_APERTURE_
# VIOLATION on every Matmul128x128* variant — the rewrite pass was
# ASYMMETRIC, preserving uniform writelane sites (writing hardware
# lane N only via `v_writelane_b32`) while rewriting divergent
# readlane sites on the same VGPR to `ds_bpermute` that reads BOTH
# lane N AND lane N + source_wave_size. The upper source-wave
# replica's lane N was never populated, so the bpermute returned
# undef, which fed into address math and produced out-of-range
# global-memory accesses.
#
# The symmetry fix (rewrite_cross_lane_divergent.cpp, this hand-off)
# makes every writelane AND every readlane site rewrite under cross-
# widening — see the "WRITELANE SYMMETRY" / "READLANE SYMMETRY"
# sections of the header comment. The asymmetric-rewrite fault path
# is closed for every Matmul128x128* variant, and the uniform-diag
# variant passes bit-exact because the pattern A=B=1.0 produces a
# position-invariant reference (K=128 everywhere), which cancels any
# residual per-source-wave layout differences.
#
# The random-input variants still show ~3% numerical errors, all
# localised to the last four rows of source wave 3's output sub-tile
# (output rows 124–127 for the single-tile case; the analogous rows
# 252–255 for the 2×2-grid case). The defect is NOT in the rewrite
# pass: the rewrite math is semantically equivalent to source
# semantics on any (val, old, src) divergence triple (proven in
# rewrite_cross_lane_divergent.cpp comments), and removing the
# rewrite entirely puts the kernel back into classifier-refusal, not
# into correctness.
#
# The residual bug was originally conjectured to live in the
# partial-EXEC WMMA → MFMA pipeline and to be cured by
# `WaveNativeProjection::emitInitialExec`'s
# `@llvm.amdgcn.init_whole_wave` preamble (hardware EXEC = -1 for the
# whole kernel body). The matmul gtests now opt in to that projection
# via `enableWaveNative=true` alongside `enableWritelaneRewrite=true`
# (see `doTestMatmul` in `tests/gfx1250_gpu_test.cpp` and
# `wave_projection.{hpp,cpp}`), and the HSACO shape confirms the
# intrinsic is reaching codegen: `s_or_saveexec_b64 s[0:1], -1` at the
# top of `matmul_kernel` and the captured original EXEC threading
# through `v_cndmask_b32` / ballot / `emitUnderExec` diamonds exactly
# as designed.
#
# The errors nevertheless remain: 490 for the single-tile case and
# 1985 for the 2×2 grid, identical row pattern (output rows 124–127 /
# 252–255) to the pre-WaveNative runs. That means the defect is NOT
# partial EXEC at the MFMA collective — the WaveNative projection
# provably fixes that symptom, and the error pattern is insensitive
# to the fix. The residual must live deeper in the WMMA → MFMA lane
# redistribution math (likely the "collect" stage in
# `wmma_lowering.cpp` for the second half of source wave 3's tile)
# or in some other bookkeeping the matmul kernel exercises that this
# investigation has not yet localised.
#
# Fixing that residual is out of scope for this hand-off. Landing the
# symmetry fix + WaveNative plumbing under WILL_FAIL keeps the
# memory-aperture-violation path closed, keeps the uniform-diag test
# as a live regression gate, lands the WaveNative projection for the
# 5 lit fixtures it was designed to unblock, and captures the
# remaining numerical issue here so the next follow-up knows exactly
# where to look (grep `wmma_lowering.cpp` for "COLLECT" and the
# per-source-wave lane-group selection at the end of `runGroupPass`).
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
set_tests_properties(Gfx1250Gpu.Matmul128x128 PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

# Two diagnostic probe patterns that sharpen the Matmul128x128 bug
# shape. Both exercise the same WMMA → MFMA redistribute/collect
# pipeline as the random-input variants above, but with inputs
# designed to isolate exactly which rows/lanes are mis-routed.
#
#   Matmul128x128_1tile_RowIdA:  A[i,k] = (i+1) * 0.001 for all k;
#     B = 1.0. Reference C[i,j] = 128 * (i+1) * 0.001, constant
#     across columns. A failing run shows output rows 124..127
#     getting contributions from A[0], A[2], A[4], A[6] for ~32
#     of the 128 K-steps (roughly ONE WMMA-call-worth of mis-
#     routed K-accumulation), arithmetic-identified as
#     "source row used = 2*(output row - 124)".
#
#   Matmul128x128_1tile_RowOnly124: A[i,k] = (i == 124 ? 1 : 0);
#     B = 1.0. Reference C[124,j] = 128 for all j; other rows = 0.
#     Failing runs show C[124,j] = 96 across all columns — exactly
#     one K-iter's contribution (32) missing for row 124. The
#     missing contribution is silently replaced with a different
#     row's A data that happens to be zero under this pattern,
#     confirming the defect is a DATA SUBSTITUTION (not a drop).
#
# Both probes confirm the defect is IN the data path (lane routing
# or A-fragment selection), not in the accumulator bookkeeping or
# store-address computation. The per-row error histogram now
# emitted by `doTestMatmul`'s error summary makes the row / column
# locality trivially visible to the next investigation.
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile_RowIdA PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
set_tests_properties(Gfx1250Gpu.Matmul128x128_1tile_RowOnly124 PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)

# Gfx1250Gpu.Softmax graduated from XFAIL to expected-pass once
# the P2 rewrite (v_permlane16 / v_permlanex16 emulation via
# ds_bpermute_b32; see the permlane16/permlanex16 row of
# hotswap/docs/wave-size-translation.md §5.3) landed on top of P5
# (DPP intrinsic lift). The
# kernel's two remaining classifier-blocked sites — a single
# `v_permlanex16_b32` and the DPP pattern — now both surface in the
# classifier trace as `[implemented]`, and the end-to-end test runs
# the raised IR on gfx942 hardware producing bit-exact softmax
# outputs (0 errors, maxErr=0.0). This XFAIL annotation has been
# removed.

# Integration vecadd via runPipelineAllKernels: hipError 719 (unspecified
# launch failure).  Single-kernel path (Gfx1250Gpu.Vecadd) works; the
# merged-HSACO linker path has a bug.
set_tests_properties(Integration.VecaddAllKernels PROPERTIES
  WILL_FAIL TRUE
  LABELS "transpiler;xfail"
)
