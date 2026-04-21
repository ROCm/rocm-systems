# Known expected failures (XFAIL).
#
# These tests run but are expected to fail.  CTest treats them as:
#   - PASS if the test exits non-zero (still failing as expected)
#   - FAIL if the test exits zero   (unexpectedly fixed -- update this file!)
#
# When you fix one of these, remove the corresponding set_tests_properties
# line so the test is expected to pass again.

# 128x128-tile matmul family — GRADUATED to expected-pass in commit 2
# of the --enable-writelane-rewrite rollout (see
# wave-size-translation.md §5.6.3 and rewrite_cross_lane_divergent.
# {hpp,cpp}). doTestMatmul in gfx1250_gpu_test.cpp now opts in to the
# rewrite, which replaces the scalar-source v_writelane with a
# per-lane `select` and the scalar-source v_readlane with an
# `llvm.amdgcn.ds.bpermute`, keeping the divergent wave_id value in a
# per-target-lane dword rather than letting the backend's implicit
# v_readfirstlane collapse source_wave[0]/source_wave[1] into one
# uniform. The previous WILL_FAIL annotations for
# Gfx1250Gpu.Matmul128x128_1tile, Gfx1250Gpu.Matmul128x128, and
# Gfx1250Gpu.Matmul128x128_1tile_UniformDiag were deleted as part of
# that cleanup — re-add them only if you intentionally turn the flag
# off in doTestMatmul, in which case the classifier will refuse the
# raise again and every Matmul128x128* test will go back to failing
# with ObstructionKind::WaveIdLiftScalarized.

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
